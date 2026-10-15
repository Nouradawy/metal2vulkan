//! Dead-global/debug cleanup and capability/extension closure.

use super::*;

/// Declare `ext` unless the module already does.
///
/// Every capability that is not core Vulkan needs its extension named alongside it, and each of
/// them wants the same idempotent push — one place to get it right rather than one copy per
/// extension.
fn require_extension(ctx: &mut Ctx, ext: &str) {
    let have = ctx
        .module
        .extensions
        .iter()
        .any(|e| matches!(e.operands.first(), Some(Operand::LiteralString(s)) if s == ext));
    if !have {
        ctx.module.extensions.push(Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString(ext.into())],
        ));
    }
}

fn needed_variable_pointer_capabilities(ctx: &Ctx) -> (bool, bool) {
    crate::spirv_variable_ptr::variable_pointer_requirement(&ctx.module)
}

/// Collect every id DEFINED in the module (globals + function defs/params/labels/instruction
/// results). Used to spot dangling debug-name / decoration references.
fn defined_ids(module: &Module) -> HashSet<Word> {
    let mut s = HashSet::new();
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            s.insert(id);
        }
    }
    for inst in &module.ext_inst_imports {
        if let Some(id) = inst.result_id {
            s.insert(id);
        }
    }
    for f in &module.functions {
        if let Some(id) = f.def.as_ref().and_then(|d| d.result_id) {
            s.insert(id);
        }
        for p in &f.parameters {
            if let Some(id) = p.result_id {
                s.insert(id);
            }
        }
        for b in &f.blocks {
            if let Some(id) = b.label.as_ref().and_then(|l| l.result_id) {
                s.insert(id);
            }
            for inst in &b.instructions {
                if let Some(id) = inst.result_id {
                    s.insert(id);
                }
            }
        }
    }
    s
}

/// Collect ids referenced from function bodies. Entry-point interface variables should be live
/// because code loads/stores/access-chains them, not because an earlier pass registered them before a
/// later raw-alias rewrite replaced every use.
pub(super) fn function_referenced_ids(module: &Module) -> HashSet<Word> {
    let mut s = HashSet::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                for op in &inst.operands {
                    if let Operand::IdRef(r) | Operand::IdScope(r) | Operand::IdMemorySemantics(r) =
                        op
                    {
                        s.insert(*r);
                    }
                }
            }
        }
    }
    s
}

/// Remove global variables that are no longer referenced by code and no longer listed on the filtered
/// entry interface. Decorations/debug names for the removed ids are swept by drop_dangling_debug.
pub(super) fn drop_dead_unreferenced_variables(
    ctx: &mut Ctx,
    function_refs: &HashSet<Word>,
    interface_ids: &HashSet<Word>,
) {
    ctx.module.types_global_values.retain(|inst| {
        if inst.class.opcode != Op::Variable {
            return true;
        }
        let Some(id) = inst.result_id else {
            return true;
        };
        function_refs.contains(&id) || interface_ids.contains(&id)
    });
}

/// Whether a non-semantic record (`OpName`, `OpDecorate`, `OpMemberName`, `OpMemberDecorate`) is
/// about one of `removed`. Every one of them names its target first.
fn targets_any(instruction: &Instruction, removed: &HashSet<Word>) -> bool {
    matches!(instruction.operands.first(), Some(Operand::IdRef(id)) if removed.contains(id))
}

/// The types whose values are opaque descriptor handles: images, samplers, sampled images, and
/// arrays of them. A pointer to one is NOT in the set — a pointer is an address, not a handle.
fn opaque_handle_types(module: &Module) -> HashSet<Word> {
    let mut handles = HashSet::new();
    // A type is defined before any array of it, so one pass in module order closes the set.
    for instruction in &module.types_global_values {
        let Some(id) = instruction.result_id else {
            continue;
        };
        let is_handle = match instruction.class.opcode {
            Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage => true,
            Op::TypeArray | Op::TypeRuntimeArray => matches!(
                instruction.operands.first(),
                Some(Operand::IdRef(element)) if handles.contains(element)
            ),
            _ => false,
        };
        if is_handle {
            handles.insert(id);
        }
    }
    handles
}

/// Whether `instruction` only produces a handle value from its operands, so that deleting it when
/// nothing reads the result changes no observable behavior.
///
/// A volatile load is the one `OpLoad` that is not pure.
fn is_pure_handle_producer(instruction: &Instruction) -> bool {
    match instruction.class.opcode {
        Op::CopyObject | Op::SampledImage | Op::Image | Op::Phi | Op::Select => true,
        Op::Load => !instruction.operands.iter().any(|operand| {
            matches!(
                operand,
                Operand::MemoryAccess(access) if access.contains(spirv::MemoryAccess::VOLATILE)
            )
        }),
        _ => false,
    }
}

/// Delete the opaque-handle values nothing consumes, together with the `OpName`/`OpDecorate`
/// records about them. Returns whether anything was removed.
///
/// Every `texture` and `sampler` parameter of an entry point becomes a `UniformConstant` variable
/// that the entry prologue loads once. When the Metal function never samples, reads, or writes that
/// parameter, the loaded handle has no consumer — yet the load is a reference to the variable, so
/// the variable, its `DescriptorSet`/`Binding` decorations, and its entry-point interface entry
/// all stay, and the module demands a descriptor the shader never touches. A consumer that builds
/// its layout from the module then finds an unbound resource at that binding on every draw.
///
/// Liveness is transitive from the consumers, not "appears as an operand": a handle is live when
/// something that is not itself a pure handle producer reads it (a sample, a store, a call
/// argument), or when a live producer reads it. That
/// collects a whole dead chain (`OpLoad` → `OpSampledImage` → `OpCopyObject`) and a dead phi cycle
/// in one pass, so no fixed-point loop is needed. A debug name or decoration about a handle is not
/// a consumer.
///
/// This is deliberately narrower than general dead-value elimination, which the pipeline does not
/// perform: only values of opaque handle type are candidates, and only the pure producers of such
/// values. Anything with an effect — a store, a call, an image read or write — is a consumer, and
/// a handle it names stays live. It also covers the synthesized placeholder descriptors
/// (`air.get_read_sampler()`, `air.get_null_texture_*()`) whose value nothing uses.
fn drop_unconsumed_handle_values(module: &mut Module) -> bool {
    let handle_types = opaque_handle_types(module);
    if handle_types.is_empty() {
        return false;
    }
    let is_candidate = |instruction: &Instruction| {
        instruction.result_id.is_some()
            && instruction
                .result_type
                .is_some_and(|ty| handle_types.contains(&ty))
            && is_pure_handle_producer(instruction)
    };
    let id_operands = |instruction: &Instruction| {
        instruction
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) => {
                    Some(*id)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };

    // Candidate result id -> the ids it reads.
    let mut producers: HashMap<Word, Vec<Word>> = HashMap::new();
    let mut pending: Vec<Word> = Vec::new();
    for function in &module.functions {
        for instruction in function.all_inst_iter() {
            match instruction.result_id {
                Some(result) if is_candidate(instruction) => {
                    producers.insert(result, id_operands(instruction));
                }
                _ => pending.extend(id_operands(instruction)),
            }
        }
    }
    if producers.is_empty() {
        return false;
    }
    // Module-scope consumers (initializers, spec-constant operations, the entry interface,
    // execution modes). Debug names and decoration targets are not consumers; the extra operands
    // of a decoration (`OpDecorateId`) can be real references, so those count.
    for section in [
        &module.entry_points,
        &module.execution_modes,
        &module.debug_string_source,
        &module.debug_module_processed,
        &module.types_global_values,
    ] {
        for instruction in section {
            pending.extend(id_operands(instruction));
        }
    }
    for instruction in &module.annotations {
        pending.extend(id_operands(instruction).into_iter().skip(1));
    }

    let mut live = HashSet::new();
    while let Some(id) = pending.pop() {
        if !live.insert(id) {
            continue;
        }
        if let Some(reads) = producers.get(&id) {
            pending.extend(reads.iter().copied().filter(|read| !live.contains(read)));
        }
    }

    let removed = producers
        .keys()
        .copied()
        .filter(|id| !live.contains(id))
        .collect::<HashSet<_>>();
    if removed.is_empty() {
        return false;
    }
    for function in &mut module.functions {
        for block in &mut function.blocks {
            block.instructions.retain(|instruction| {
                !(is_candidate(instruction)
                    && instruction
                        .result_id
                        .is_some_and(|id| removed.contains(&id)))
            });
        }
    }
    module
        .annotations
        .retain(|instruction| !targets_any(instruction, &removed));
    module
        .debug_names
        .retain(|instruction| !targets_any(instruction, &removed));
    true
}

/// Drop every global `OpVariable` no instruction references, together with its entry-point interface
/// entry, its decorations, and its debug names. Returns whether anything was removed.
///
/// A descriptor variable is referenced by the entry prologue's `OpLoad` even when nothing consumes
/// the loaded handle, so the rule first retracts the handle values nothing consumes
/// ([`drop_unconsumed_handle_values`]); only then is "no instruction references it" true of a
/// texture or sampler parameter the shader never uses.
///
/// `finalize` applies this rule once, against the bodies as they stand when it builds the entry
/// point. Instruction-deleting work runs after it at two boundaries — the pointer and access-chain
/// closures at the end of `transform`, and dead-value elimination plus constant-CFG pruning in
/// `finish_module` — and when one of those deletes a variable's last use, nothing downstream can
/// remove the variable: `gc_dead_globals` roots liveness AT the entry-point interface, so the
/// interface entry keeps the variable alive and the variable justifies the interface entry. The
/// module then ships an `OpVariable` with a `DescriptorSet`/`Binding` no instruction touches, and a
/// consumer building a descriptor-set layout from the interface has to satisfy a binding the shader
/// never reads.
///
/// Re-applying the rule after the last instruction-deleting step breaks that cycle. It can only
/// shrink the interface — an id kept here is one an instruction still names — so SPIR-V 1.4's
/// requirement that the interface list every global the entry point references is preserved.
pub(crate) fn drop_unreferenced_global_variables(module: &mut Module) -> bool {
    /// `OpEntryPoint` operands: execution model, entry function id, name, then the interface ids.
    const INTERFACE_START: usize = 3;

    let mut changed = drop_unconsumed_handle_values(module);
    let referenced = function_referenced_ids(module);
    for entry_point in &mut module.entry_points {
        let before = entry_point.operands.len();
        entry_point.operands = std::mem::take(&mut entry_point.operands)
            .into_iter()
            .enumerate()
            .filter(|(index, operand)| match operand {
                Operand::IdRef(id) if *index >= INTERFACE_START => referenced.contains(id),
                _ => true,
            })
            .map(|(_, operand)| operand)
            .collect();
        changed |= entry_point.operands.len() != before;
    }
    let removed = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::Variable)
        .filter_map(|instruction| instruction.result_id)
        .filter(|id| !referenced.contains(id))
        .collect::<HashSet<_>>();
    if removed.is_empty() {
        return changed;
    }
    module.types_global_values.retain(|instruction| {
        instruction
            .result_id
            .is_none_or(|id| !removed.contains(&id))
    });
    module
        .annotations
        .retain(|instruction| !targets_any(instruction, &removed));
    module
        .debug_names
        .retain(|instruction| !targets_any(instruction, &removed));
    true
}

/// Remove OpName/OpDecorate/OpMemberDecorate whose target id is not defined anywhere.
pub(crate) fn drop_dangling_debug(module: &mut Module) {
    let defined = defined_ids(module);
    let keep = |inst: &Instruction| -> bool {
        match inst.operands.first() {
            Some(Operand::IdRef(id)) => defined.contains(id),
            _ => true,
        }
    };
    module.debug_names.retain(&keep);
    module.annotations.retain(&keep);
}

/// Iteratively drop types/global-values whose result id is referenced nowhere else in the module
/// (instruction operands, result-types, decorations, entry-point interface, function bodies). Keeps
/// variables referenced by the entry interface. Repeats to a fixpoint so chains die cleanly.
pub(super) fn gc_dead_globals(ctx: &mut Ctx) {
    let collect = |inst: &Instruction, live: &mut HashSet<Word>| {
        if let Some(ty) = inst.result_type {
            live.insert(ty);
        }
        for operand in &inst.operands {
            if let Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) =
                operand
            {
                live.insert(*id);
            }
        }
    };

    // Executable/module linkage, semantic decorations, and typed sidecars own liveness. Debug names
    // follow those definitions; treating names as roots retains dead constants (including pointer
    // nulls that Vulkan's Logical addressing model cannot represent) forever.
    let mut live = HashSet::new();
    live.extend(
        ctx.emit_sidecar
            .local_pointer_field_stores
            .iter()
            .map(|fact| fact.id),
    );
    live.extend(ctx.emit_sidecar.buffer_root_source_types.values().copied());
    for section in [&ctx.module.entry_points, &ctx.module.execution_modes] {
        for instruction in section {
            collect(instruction, &mut live);
        }
    }
    for instruction in &ctx.module.annotations {
        // ArrayStride is metadata owned by its target type, not an independent liveness root. A
        // late BDA candidate can leave a decorated physical pointer type after its executable
        // address path is eliminated; rooting the type from this decoration would retain an
        // otherwise-dead capability-requiring declaration in a Logical module.
        if instruction.operands.get(1) == Some(&Operand::Decoration(spirv::Decoration::ArrayStride))
        {
            continue;
        }
        collect(instruction, &mut live);
    }
    for function in &ctx.module.functions {
        if let Some(definition) = &function.def {
            collect(definition, &mut live);
        }
        for parameter in &function.parameters {
            collect(parameter, &mut live);
        }
        for block in &function.blocks {
            if let Some(label) = &block.label {
                collect(label, &mut live);
            }
            for instruction in &block.instructions {
                collect(instruction, &mut live);
            }
        }
    }

    let definitions = ctx
        .module
        .types_global_values
        .iter()
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
        .collect::<HashMap<_, _>>();
    let mut pending = live.iter().copied().collect::<Vec<_>>();
    while let Some(id) = pending.pop() {
        let Some(definition) = definitions.get(&id) else {
            continue;
        };
        let mut dependencies = HashSet::new();
        collect(definition, &mut dependencies);
        for dependency in dependencies {
            if live.insert(dependency) {
                pending.push(dependency);
            }
        }
    }

    ctx.module
        .types_global_values
        .retain(|instruction| instruction.result_id.is_none_or(|id| live.contains(&id)));
    drop_dangling_debug(&mut ctx.module);
}

/// Remove `OpCapability Int64` when no `OpTypeInt 64` survives (i.e. no genuine 64-bit-int type is
/// used anywhere after access-chain narrowing + dead-global gc). Leaving a capability whose feature is
/// unused is legal for spirv-val, but the Int64 declaration is what cues NVIDIA's compiler down the
/// 64-bit path that crashes; dropping it keeps the module strictly 32-bit. If genuine i64 math remains
/// the type is still present and we leave the capability (and `add_needed_capabilities` would re-add it).
pub(super) fn drop_unused_int64_capability(ctx: &mut Ctx) {
    let has_int64_type = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::LiteralBit32(64))
    });
    if has_int64_type {
        return;
    }
    ctx.module.capabilities.retain(|c| {
        !matches!(
            c.operands.first(),
            Some(Operand::Capability(spirv::Capability::Int64))
        )
    });
}

pub(super) fn drop_unused_variable_pointer_capabilities(ctx: &mut Ctx) -> (bool, bool) {
    crate::spirv_variable_ptr::lower_zero_base_storage_buffer_ptr_access_chains(&mut ctx.module);
    crate::spirv_variable_ptr::rewrite_storage_buffer_atomic_scopes(&mut ctx.module);
    let (needs_storage_buffer, needs_other) = needed_variable_pointer_capabilities(ctx);
    ctx.module
        .capabilities
        .retain(|c| match c.operands.first() {
            Some(Operand::Capability(spirv::Capability::VariablePointersStorageBuffer)) => {
                needs_storage_buffer
            }
            Some(Operand::Capability(spirv::Capability::VariablePointers)) => needs_other,
            _ => true,
        });
    (needs_storage_buffer, needs_other)
}

pub(super) fn add_needed_capabilities(ctx: &mut Ctx, variable_pointer_requirements: (bool, bool)) {
    use spirv::Capability;
    let mut want: Vec<Capability> = vec![];
    if ctx.module.entry_points.iter().any(|instruction| {
        instruction.operands.first()
            == Some(&Operand::ExecutionModel(
                spirv::ExecutionModel::TessellationEvaluation,
            ))
    }) {
        want.push(Capability::Tessellation);
    }
    let has_demote = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::DemoteToHelperInvocation);
    if has_demote {
        want.push(Capability::DemoteToHelperInvocation);
    }
    let has_sampled_1d = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage && i.operands.get(1) == Some(&Operand::Dim(Dim::Dim1D))
    });
    let has_storage_1d = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage
            && i.operands.get(1) == Some(&Operand::Dim(Dim::Dim1D))
            && i.operands.get(5) == Some(&Operand::LiteralBit32(2))
    });
    if has_sampled_1d {
        want.push(Capability::Sampled1D);
    }
    if has_storage_1d {
        want.push(Capability::Image1D);
    }
    let has_buffer_image = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage && i.operands.get(1) == Some(&Operand::Dim(Dim::DimBuffer))
    });
    let has_storage_buffer_image = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage
            && i.operands.get(1) == Some(&Operand::Dim(Dim::DimBuffer))
            && i.operands.get(5) == Some(&Operand::LiteralBit32(2))
    });
    if has_buffer_image {
        want.push(Capability::SampledBuffer);
    }
    if has_storage_buffer_image {
        want.push(Capability::ImageBuffer);
    }
    let has_input_attachment_type = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage
            && i.operands.get(1) == Some(&Operand::Dim(Dim::DimSubpassData))
    });
    let has_input_attachment_decor = ctx.module.annotations.iter().any(|i| {
        i.class.opcode == Op::Decorate
            && i.operands.get(1) == Some(&Operand::Decoration(Decoration::InputAttachmentIndex))
    });
    if has_input_attachment_type || has_input_attachment_decor {
        want.push(Capability::InputAttachment);
    }
    let has_viewport_index = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::ViewportIndex))
    });
    if has_viewport_index {
        want.push(Capability::ShaderViewportIndex);
        if let Some(header) = ctx.module.header.as_mut() {
            let version = header.version();
            if version < (1, 5) {
                header.set_version(1, 5);
            }
        }
    }
    let has_draw_parameters = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && matches!(
                instruction.operands.get(2),
                Some(&Operand::BuiltIn(
                    BuiltIn::BaseInstance | BuiltIn::BaseVertex
                ))
            )
    });
    if has_draw_parameters {
        // Core since SPIR-V 1.3; below it the capability needs SPV_KHR_shader_draw_parameters.
        want.push(Capability::DrawParameters);
        if let Some(header) = ctx.module.header.as_mut() {
            let version = header.version();
            if version < (1, 3) {
                header.set_version(1, 3);
            }
        }
    }
    let has_layer = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::Layer))
    });
    if has_layer {
        want.push(Capability::ShaderLayer);
        if let Some(header) = ctx.module.header.as_mut() {
            let version = header.version();
            if version < (1, 5) {
                header.set_version(1, 5);
            }
        }
    }
    let has_clip_distance = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::ClipDistance))
    });
    if has_clip_distance {
        want.push(Capability::ClipDistance);
    }
    let has_primitive_id = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::PrimitiveId))
    });
    if has_primitive_id {
        want.push(Capability::Geometry);
    }
    let has_sample_id = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::SampleId))
    });
    // `Sample` on an Input variable is per-sample interpolation (Metal `[[sample_perspective]]`),
    // which requests sample-rate shading just as reading `SampleId` does.
    let has_sample_interpolation = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::Sample))
    });
    if has_sample_id || has_sample_interpolation {
        want.push(Capability::SampleRateShading);
    }
    // `BaryCoordKHR` / `BaryCoordNoPerspKHR` are `[[barycentric_coord]]`; both come from the
    // fragment-barycentric extension rather than core Vulkan.
    let has_barycentric = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && matches!(
                instruction.operands.get(2),
                Some(&Operand::BuiltIn(BuiltIn::BaryCoordKHR))
                    | Some(&Operand::BuiltIn(BuiltIn::BaryCoordNoPerspKHR))
            )
    });
    if has_barycentric {
        want.push(Capability::FragmentBarycentricKHR);
    }
    let has_stencil_export = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::FragStencilRefEXT))
    });
    if has_stencil_export {
        want.push(Capability::StencilExportEXT);
    }
    // Texture query opcodes need the ImageQuery capability.
    let has_query = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::ImageQuerySize
                    | Op::ImageQuerySizeLod
                    | Op::ImageQueryLod
                    | Op::ImageQueryLevels
                    | Op::ImageQuerySamples
            )
        });
    if has_query {
        want.push(Capability::ImageQuery);
    }
    let has_group_shuffle = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformShuffle | Op::GroupNonUniformShuffleXor
            )
        });
    if has_group_shuffle {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformShuffle);
    }
    let has_group_shuffle_relative = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformShuffleUp | Op::GroupNonUniformShuffleDown
            )
        });
    if has_group_shuffle_relative {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformShuffleRelative);
    }
    let has_group_elect = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::GroupNonUniformElect);
    if has_group_elect {
        want.push(Capability::GroupNonUniform);
    }
    let has_group_vote = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformAll | Op::GroupNonUniformAny | Op::GroupNonUniformAllEqual
            )
        });
    if has_group_vote {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformVote);
    }
    let has_group_ballot = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformBallot
                    | Op::GroupNonUniformBallotBitExtract
                    | Op::GroupNonUniformBallotBitCount
                    | Op::GroupNonUniformBallotFindLSB
                    | Op::GroupNonUniformBallotFindMSB
            )
        });
    if has_group_ballot {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformBallot);
    }
    let has_group_broadcast = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformBroadcast | Op::GroupNonUniformBroadcastFirst
            )
        });
    if has_group_broadcast {
        // BroadcastFirst (and Broadcast) are gated by the Ballot capability in core SPIR-V.
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformBallot);
    }
    let has_subgroup_invocation_builtin = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.iter().any(|operand| {
                matches!(
                    operand,
                    Operand::BuiltIn(spirv::BuiltIn::SubgroupLocalInvocationId)
                )
            })
    });
    if has_subgroup_invocation_builtin {
        want.push(Capability::GroupNonUniform);
    }
    let has_group_arithmetic = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformIAdd
                    | Op::GroupNonUniformFAdd
                    | Op::GroupNonUniformSMin
                    | Op::GroupNonUniformUMin
                    | Op::GroupNonUniformFMin
                    | Op::GroupNonUniformSMax
                    | Op::GroupNonUniformUMax
                    | Op::GroupNonUniformFMax
                    | Op::GroupNonUniformBitwiseAnd
                    | Op::GroupNonUniformBitwiseOr
                    | Op::GroupNonUniformBitwiseXor
            )
        });
    if has_group_arithmetic {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformArithmetic);
    }
    // A `ClusteredReduce` group operation (emitted by the M-D2 `TransformOptions::simd_cluster32`
    // simd lowering) additionally needs the `GroupNonUniformClustered` capability.
    let has_clustered_reduce = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            i.operands.iter().any(|o| {
                matches!(
                    o,
                    Operand::GroupOperation(spirv::GroupOperation::ClusteredReduce)
                )
            })
        });
    if has_clustered_reduce {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformClustered);
    }
    let has_atomic_fadd = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::AtomicFAddEXT);
    if has_atomic_fadd {
        want.push(Capability::AtomicFloat32AddEXT);
    }
    // Width-based scalar capabilities: a `half` (OpTypeFloat 16) needs Float16; an 8-/16-bit int
    // needs Int8/Int16. Lowering can synthesize narrow constants and types (saturate edges, FConvert
    // results, and flat-buffer extracts), so assert the corresponding capabilities here.
    let int_width = |w: u32| {
        ctx.module.types_global_values.iter().any(|i| {
            i.class.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::LiteralBit32(w))
        })
    };
    let has_half = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeFloat && i.operands.first() == Some(&Operand::LiteralBit32(16))
    });
    if has_half {
        want.push(Capability::Float16);
    }
    if int_width(8) {
        want.push(Capability::Int8);
    }
    if int_width(16) {
        want.push(Capability::Int16);
    }
    if int_width(64) {
        want.push(Capability::Int64);
    }
    let (has_storage_buffer_pointer_merge, has_other_pointer_merge) = variable_pointer_requirements;
    if has_storage_buffer_pointer_merge {
        want.push(Capability::VariablePointersStorageBuffer);
    }
    if has_other_pointer_merge {
        want.push(Capability::VariablePointers);
    }
    for cap in &want {
        let already = ctx.module.capabilities.iter().any(|c| {
            matches!(c.operands.first(), Some(Operand::Capability(existing)) if existing == cap)
        });
        if !already {
            ctx.module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(*cap)],
            ));
        }
    }
    // DemoteToHelperInvocation needs its SPIR-V extension declared (core only from SPIR-V 1.6).
    if want.contains(&Capability::DemoteToHelperInvocation) {
        let ext = "SPV_EXT_demote_to_helper_invocation";
        require_extension(ctx, ext);
    }
    if want.contains(&Capability::AtomicFloat32AddEXT) {
        let ext = "SPV_EXT_shader_atomic_float_add";
        require_extension(ctx, ext);
    }
    if want.contains(&Capability::FragmentBarycentricKHR) {
        let ext = "SPV_KHR_fragment_shader_barycentric";
        require_extension(ctx, ext);
    }
    if want.contains(&Capability::StencilExportEXT) {
        let ext = "SPV_EXT_shader_stencil_export";
        require_extension(ctx, ext);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::Module;
    use crate::spirv_module::ModuleHeader;

    /// A module whose entry point lists two descriptor variables while only one is still loaded.
    /// The second is what a pipeline boundary strands when a later rewrite deletes its last use.
    fn module_with_one_stranded_descriptor() -> Module {
        use crate::spirv_module::{Block, Function};
        let (uint, pointer, live, stranded, entry, loaded) = (1, 2, 3, 4, 5, 6);
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(7));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uint),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(pointer),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(uint),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(pointer),
                Some(live),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            Instruction::new(
                Op::Variable,
                Some(pointer),
                Some(stranded),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        for (variable, binding) in [(live, 0), (stranded, 1)] {
            module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(variable),
                    Operand::Decoration(spirv::Decoration::Binding),
                    Operand::LiteralBit32(binding),
                ],
            ));
            module.debug_names.push(Instruction::new(
                Op::Name,
                None,
                None,
                vec![
                    Operand::IdRef(variable),
                    Operand::LiteralString(format!("buffer{binding}")),
                ],
            ));
        }
        module.entry_points.push(Instruction::new(
            Op::EntryPoint,
            None,
            None,
            vec![
                Operand::ExecutionModel(spirv::ExecutionModel::GLCompute),
                Operand::IdRef(entry),
                Operand::LiteralString("main".to_string()),
                Operand::IdRef(live),
                Operand::IdRef(stranded),
            ],
        ));
        let mut function = Function::new();
        let mut block = Block::new();
        block.instructions.push(Instruction::new(
            Op::Load,
            Some(uint),
            Some(loaded),
            vec![Operand::IdRef(live)],
        ));
        function.blocks.push(block);
        module.functions.push(function);
        module
    }

    /// The ids `module_with_one_stranded_descriptor` uses, in the order it names them.
    const LIVE_VARIABLE: Word = 3;
    const STRANDED_VARIABLE: Word = 4;

    fn variable_ids(module: &Module) -> HashSet<Word> {
        module
            .types_global_values
            .iter()
            .filter(|instruction| instruction.class.opcode == Op::Variable)
            .filter_map(|instruction| instruction.result_id)
            .collect()
    }

    fn ids_named_by(records: &[Instruction]) -> HashSet<Word> {
        records
            .iter()
            .filter_map(|instruction| match instruction.operands.first() {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_variable_no_instruction_references_leaves_with_its_interface_entry_and_records() {
        let mut module = module_with_one_stranded_descriptor();

        assert!(drop_unreferenced_global_variables(&mut module));

        assert_eq!(variable_ids(&module), HashSet::from([LIVE_VARIABLE]));
        let interface = module.entry_points[0].operands[3..]
            .iter()
            .filter_map(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .collect::<HashSet<_>>();
        assert_eq!(interface, HashSet::from([LIVE_VARIABLE]));
        assert!(!ids_named_by(&module.annotations).contains(&STRANDED_VARIABLE));
        assert!(!ids_named_by(&module.debug_names).contains(&STRANDED_VARIABLE));
        // The surviving descriptor keeps everything that describes it.
        assert!(ids_named_by(&module.annotations).contains(&LIVE_VARIABLE));
        assert!(ids_named_by(&module.debug_names).contains(&LIVE_VARIABLE));
    }

    #[test]
    fn the_sweep_reaches_a_fixed_point_in_one_pass() {
        let mut module = module_with_one_stranded_descriptor();
        assert!(drop_unreferenced_global_variables(&mut module));
        // Nothing left to remove, so a pipeline that runs it at more than one boundary pays a scan
        // and changes nothing.
        assert!(!drop_unreferenced_global_variables(&mut module));
    }

    #[test]
    fn gc_keeps_local_pointer_store_sentinel_rooted_by_typed_sidecar() {
        let ulong = 1;
        let sentinel = 2;
        let dead = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(4));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(ulong),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(sentinel), vec![]),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(dead), vec![]),
        ];
        let mut ctx = Ctx::new(module);
        ctx.emit_sidecar.local_pointer_field_stores.push(
            crate::emit_sidecar::LocalPointerFieldStore {
                id: sentinel,
                source: 99,
                root: 98,
                indices: Vec::new(),
            },
        );

        gc_dead_globals(&mut ctx);

        let ids = ctx
            .module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        assert!(ids.contains(&ulong));
        assert!(ids.contains(&sentinel));
        assert!(!ids.contains(&dead));
    }

    #[test]
    fn gc_drops_dead_global_even_when_debug_metadata_names_it() {
        let byte = 1;
        let pointer = 2;
        let null = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(4));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(byte),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(pointer),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(byte),
                ],
            ),
            Instruction::new(Op::ConstantNull, Some(pointer), Some(null), vec![]),
        ];
        module.debug_names.push(Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(null),
                Operand::LiteralString("dead pointer".to_string()),
            ],
        ));
        let mut ctx = Ctx::new(module);

        gc_dead_globals(&mut ctx);

        assert!(ctx.module.types_global_values.is_empty());
        assert!(ctx.module.debug_names.is_empty());
    }

    #[test]
    fn gc_drops_dead_pointer_type_owned_only_by_array_stride() {
        let byte = 1;
        let pointer = 2;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(3));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(byte),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(pointer),
                vec![
                    Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                    Operand::IdRef(byte),
                ],
            ),
        ];
        module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(pointer),
                Operand::Decoration(spirv::Decoration::ArrayStride),
                Operand::LiteralBit32(1),
            ],
        ));
        let mut ctx = Ctx::new(module);

        gc_dead_globals(&mut ctx);

        assert!(ctx.module.types_global_values.is_empty());
        assert!(ctx.module.annotations.is_empty());
    }
}

#[cfg(test)]
mod unconsumed_handle_tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    const FLOAT: Word = 1;
    const IMAGE: Word = 2;
    const SAMPLER: Word = 3;
    const SAMPLED_IMAGE: Word = 4;
    const IMAGE_POINTER: Word = 5;
    const SAMPLER_POINTER: Word = 6;
    const COORD: Word = 7;
    const IMAGE_ARRAY: Word = 8;
    /// The first id the fixtures hand out to variables.
    const FIRST_VARIABLE: Word = 10;
    const ENTRY: Word = 40;

    fn handle_module(descriptors: &[(Word, Word)]) -> Module {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeFloat,
                None,
                Some(FLOAT),
                vec![Operand::LiteralBit32(32)],
            ),
            Instruction::new(
                Op::TypeImage,
                None,
                Some(IMAGE),
                vec![
                    Operand::IdRef(FLOAT),
                    Operand::Dim(spirv::Dim::Dim2D),
                    Operand::LiteralBit32(0),
                    Operand::LiteralBit32(0),
                    Operand::LiteralBit32(0),
                    Operand::LiteralBit32(1),
                    Operand::ImageFormat(spirv::ImageFormat::Unknown),
                ],
            ),
            Instruction::new(Op::TypeSampler, None, Some(SAMPLER), vec![]),
            Instruction::new(
                Op::TypeSampledImage,
                None,
                Some(SAMPLED_IMAGE),
                vec![Operand::IdRef(IMAGE)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(IMAGE_POINTER),
                vec![
                    Operand::StorageClass(StorageClass::UniformConstant),
                    Operand::IdRef(IMAGE),
                ],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(SAMPLER_POINTER),
                vec![
                    Operand::StorageClass(StorageClass::UniformConstant),
                    Operand::IdRef(SAMPLER),
                ],
            ),
            Instruction::new(Op::ConstantNull, Some(FLOAT), Some(COORD), vec![]),
            Instruction::new(
                Op::TypeRuntimeArray,
                None,
                Some(IMAGE_ARRAY),
                vec![Operand::IdRef(IMAGE)],
            ),
        ];
        let mut entry_operands = vec![
            Operand::ExecutionModel(spirv::ExecutionModel::Fragment),
            Operand::IdRef(ENTRY),
            Operand::LiteralString("main".to_string()),
        ];
        for (index, (variable, pointer)) in descriptors.iter().enumerate() {
            module.types_global_values.push(Instruction::new(
                Op::Variable,
                Some(*pointer),
                Some(*variable),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(*variable),
                    Operand::Decoration(spirv::Decoration::Binding),
                    Operand::LiteralBit32(index as u32),
                ],
            ));
            module.debug_names.push(Instruction::new(
                Op::Name,
                None,
                None,
                vec![
                    Operand::IdRef(*variable),
                    Operand::LiteralString(format!("resource{index}")),
                ],
            ));
            entry_operands.push(Operand::IdRef(*variable));
        }
        module
            .entry_points
            .push(Instruction::new(Op::EntryPoint, None, None, entry_operands));
        module
    }

    fn load(ty: Word, result: Word, variable: Word) -> Instruction {
        Instruction::new(
            Op::Load,
            Some(ty),
            Some(result),
            vec![Operand::IdRef(variable)],
        )
    }

    fn with_body(mut module: Module, instructions: Vec<Instruction>) -> Module {
        let mut block = Block::new();
        block.instructions = instructions;
        let mut function = Function::new();
        function.blocks.push(block);
        module.functions.push(function);
        module
    }

    fn body_results(module: &Module) -> HashSet<Word> {
        module
            .functions
            .iter()
            .flat_map(|function| function.blocks.iter())
            .flat_map(|block| block.instructions.iter())
            .filter_map(|instruction| instruction.result_id)
            .collect()
    }

    fn variables(module: &Module) -> HashSet<Word> {
        module
            .types_global_values
            .iter()
            .filter(|instruction| instruction.class.opcode == Op::Variable)
            .filter_map(|instruction| instruction.result_id)
            .collect()
    }

    fn interface(module: &Module) -> HashSet<Word> {
        module.entry_points[0].operands[3..]
            .iter()
            .filter_map(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .collect()
    }

    fn records_about(records: &[Instruction]) -> HashSet<Word> {
        records
            .iter()
            .filter_map(|instruction| match instruction.operands.first() {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .collect()
    }

    /// Two textures and two samplers; only the first of each pair is sampled. The second pair is
    /// loaded, combined, copied, and merged by a phi that feeds itself -- a chain and a cycle no
    /// instruction outside them reads.
    #[test]
    fn a_dead_handle_chain_takes_its_variable_binding_name_and_interface_entry() {
        let (tex0, tex1, samp0, samp1) = (10, 11, 12, 13);
        let module = handle_module(&[
            (tex0, IMAGE_POINTER),
            (tex1, IMAGE_POINTER),
            (samp0, SAMPLER_POINTER),
            (samp1, SAMPLER_POINTER),
        ]);
        let (image0, sampler0, combined0, sample) = (20, 21, 22, 23);
        let (image1, sampler1, combined1, copy, phi) = (24, 25, 26, 27, 28);
        let mut module = with_body(
            module,
            vec![
                load(IMAGE, image0, tex0),
                load(SAMPLER, sampler0, samp0),
                Instruction::new(
                    Op::SampledImage,
                    Some(SAMPLED_IMAGE),
                    Some(combined0),
                    vec![Operand::IdRef(image0), Operand::IdRef(sampler0)],
                ),
                Instruction::new(
                    Op::ImageSampleImplicitLod,
                    Some(FLOAT),
                    Some(sample),
                    vec![Operand::IdRef(combined0), Operand::IdRef(COORD)],
                ),
                load(IMAGE, image1, tex1),
                load(SAMPLER, sampler1, samp1),
                Instruction::new(
                    Op::SampledImage,
                    Some(SAMPLED_IMAGE),
                    Some(combined1),
                    vec![Operand::IdRef(image1), Operand::IdRef(sampler1)],
                ),
                Instruction::new(
                    Op::CopyObject,
                    Some(SAMPLED_IMAGE),
                    Some(copy),
                    vec![Operand::IdRef(combined1)],
                ),
                Instruction::new(
                    Op::Phi,
                    Some(IMAGE),
                    Some(phi),
                    vec![
                        Operand::IdRef(image1),
                        Operand::IdRef(ENTRY),
                        Operand::IdRef(phi),
                        Operand::IdRef(ENTRY),
                    ],
                ),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        );

        assert!(drop_unreferenced_global_variables(&mut module));

        assert_eq!(
            body_results(&module),
            HashSet::from([image0, sampler0, combined0, sample])
        );
        assert_eq!(variables(&module), HashSet::from([tex0, samp0]));
        assert_eq!(interface(&module), HashSet::from([tex0, samp0]));
        assert_eq!(
            records_about(&module.annotations),
            HashSet::from([tex0, samp0])
        );
        assert_eq!(
            records_about(&module.debug_names),
            HashSet::from([tex0, samp0])
        );
        // Nothing left to remove.
        assert!(!drop_unreferenced_global_variables(&mut module));
    }

    /// A name or a decoration about a handle is not a use of it.
    #[test]
    fn a_name_and_a_decoration_on_the_dead_load_do_not_keep_it() {
        let tex = FIRST_VARIABLE;
        let image = 20;
        let mut module = with_body(
            handle_module(&[(tex, IMAGE_POINTER)]),
            vec![
                load(IMAGE, image, tex),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        );
        module.debug_names.push(Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(image),
                Operand::LiteralString("texture".to_string()),
            ],
        ));
        module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(image),
                Operand::Decoration(spirv::Decoration::NonUniform),
            ],
        ));

        assert!(drop_unreferenced_global_variables(&mut module));

        assert!(body_results(&module).is_empty());
        assert!(variables(&module).is_empty());
        assert!(interface(&module).is_empty());
        assert!(module.annotations.is_empty());
        assert!(module.debug_names.is_empty());
    }

    /// Every way a handle can be consumed keeps its load and its descriptor, and a handle array
    /// loaded whole is as removable as a single handle once nothing reads it.
    #[test]
    fn a_consumed_handle_keeps_its_load_and_variable() {
        let (written, passed, stored, volatile, array) = (10, 11, 12, 13, 14);
        let (helper, private, array_pointer) = (50, 51, 52);
        let mut module = handle_module(&[
            (written, IMAGE_POINTER),
            (passed, IMAGE_POINTER),
            (stored, IMAGE_POINTER),
            (volatile, IMAGE_POINTER),
            (array, array_pointer),
        ]);
        // The array pointer type follows the array type it points at and precedes the variables.
        module.types_global_values.insert(
            8,
            Instruction::new(
                Op::TypePointer,
                None,
                Some(array_pointer),
                vec![
                    Operand::StorageClass(StorageClass::UniformConstant),
                    Operand::IdRef(IMAGE_ARRAY),
                ],
            ),
        );
        let mut module = with_body(
            module,
            vec![
                load(IMAGE, 20, written),
                Instruction::new(
                    Op::ImageWrite,
                    None,
                    None,
                    vec![
                        Operand::IdRef(20),
                        Operand::IdRef(COORD),
                        Operand::IdRef(COORD),
                    ],
                ),
                load(IMAGE, 21, passed),
                Instruction::new(
                    Op::FunctionCall,
                    Some(FLOAT),
                    Some(22),
                    vec![Operand::IdRef(helper), Operand::IdRef(21)],
                ),
                load(IMAGE, 23, stored),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(private), Operand::IdRef(23)],
                ),
                // An unused volatile load is an access the shader asked for.
                Instruction::new(
                    Op::Load,
                    Some(IMAGE),
                    Some(24),
                    vec![
                        Operand::IdRef(volatile),
                        Operand::MemoryAccess(spirv::MemoryAccess::VOLATILE),
                    ],
                ),
                load(IMAGE_ARRAY, 25, array),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        );

        assert!(drop_unreferenced_global_variables(&mut module));

        assert_eq!(
            body_results(&module),
            HashSet::from([20, 21, 22, 23, 24]),
            "only the unread whole-array load goes"
        );
        assert_eq!(
            variables(&module),
            HashSet::from([written, passed, stored, volatile])
        );
    }

    /// Only handles are candidates: an unused load of ordinary data stays exactly as it was.
    #[test]
    fn an_unused_load_of_a_non_handle_is_left_alone() {
        let buffer = FIRST_VARIABLE;
        let mut module = handle_module(&[]);
        module.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(60),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(FLOAT),
            ],
        ));
        module.types_global_values.push(Instruction::new(
            Op::Variable,
            Some(60),
            Some(buffer),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ));
        let mut module = with_body(
            module,
            vec![
                load(FLOAT, 20, buffer),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        );

        assert!(!drop_unconsumed_handle_values(&mut module));

        assert_eq!(body_results(&module), HashSet::from([20]));
    }
}
