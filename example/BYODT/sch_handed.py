from tvm import relax, tir, IRModule

def _find_compute_blocks(sch: tir.Schedule):
    """Find compute blocks in the TIR schedule."""
    block_candidates = [
        "compute", "T_add", "T_relu", "T_multiply", "T_divide", 
        "T_subtract", "T_maximum", "T_minimum", 
        "conv2d", "conv2d_nchw", "conv2d_nhwc",
        "dense", "matmul","T_cast", "T_exp", "T_sqrt", "T_tanh", "T_sigmoid"
    ]
    
    blocks_found = []
    
    for block_name in block_candidates:
        try:
            sch.get_block(block_name)
            blocks_found.append(block_name)
        except:
            continue
    
    return blocks_found


def _parallelize_block(sch: tir.Schedule, block_name, func_name):
    """Add parallel directive to specified block"""
    try:
        print(f"Processing block: {block_name} in function {func_name}")
        block = sch.get_block(block_name)
        loops = sch.get_loops(block)
        
        if not loops:
            print(f"No loops found in block: {block_name}")
            return False
            
        print(f"Found {len(loops)} loops")
        

        block_success_count = 0

        for i, loop in enumerate(loops):
            try:
                loop_stmt = sch.get(loop)
                extent = loop_stmt.extent.value if hasattr(loop_stmt.extent, 'value') else None
                print(f"Analyzing loop {i}: {loop_stmt.loop_var}, range: {extent}")
                
                if extent is not None and extent <= 4:
                    print(f"Loop {loop_stmt.loop_var} is too small ({extent}), skipping")
                    continue
                    
                sch.parallel(loop)
                print(f"Parallelized loop: {loop_stmt.loop_var}, range: {extent}")
                block_success_count += 1
                break  # Only parallelize the first suitable loop
                
            except Exception as e:
                print(f"Error processing loop {i} in block {block_name}: {e}")
        
        result_msg = f"{'Success' if block_success_count > 0 else '❌ No success'} parallelizing results: {block_success_count}/1 loops in block {block_name} of function {func_name}"
        print(f"{result_msg}")
        return block_success_count > 0
        
    except Exception as e:
        print(f"Error processing block {block_name}: {e}")
        return False


def _process_function(sch, func_name):
    """Process parallelization for a single function"""
    try:
        print(f"\nProcessing function: {func_name}")
        sch.work_on(func_name)
        # Find all blocks in the function
        block_candidates = _find_compute_blocks(sch)

        if not block_candidates:
            print(f"No compute blocks found in function: {func_name}")
            return False
            
        func_success_count = 0
        # Add parallel directives to each found block
        for block_name in block_candidates:
            if _parallelize_block(sch, block_name, func_name):
                func_success_count += 1
                
        print(f"Function {func_name} result: {' Success' if func_success_count > 0 else '❌ Failed'} ({func_success_count}/{len(block_candidates)} blocks)")
        return func_success_count > 0
        
    except Exception as e:
        print(f"Error working on function {func_name}: {e}")
        return False


def add_parallel_directives_to_all_functions(mod):
    """
    Find all TIR functions in the module and add parallelization to them
    """
    sch = tir.Schedule(mod)
    # Find all TIR functions in the module
    # print(mod.functions.items())
    tir_functions = []
    for gvar, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            tir_functions.append(gvar.name_hint)
    print(f"Find {len(tir_functions)} TIR functions:")

    for func_name in tir_functions:
        print(f" - {func_name}")

    total_success_count = 0
    
    for func_name in tir_functions:
        if _process_function(sch, func_name):
            total_success_count += 1
            
    print(f"\nParallelization complete: {total_success_count}/{len(tir_functions)} functions successful")
    return sch.mod