## Simulate loop extrusion by loop extruding factors and analyse the resulting structures of loops.

- simlef.pyx - simulate loop extrusion in 1d with simulate()

- looptools.py - analyse the structure of extruded loops:
    - build trees of nested loops with get_parent_loops()
    - find root loops with get_roots()
    - identify the positions between the root loops with get_backbone()
    - identify stacked LEFs with stack_lefs()
    
- loopviz.py - visualize loops with plot_lefs()

- loopviz_circular.py - visualize loops on the genome with plot_circular_genome_with_loops()

- looplib_bacterial_gal folder:
    - bacterial_no_bypassing.pyx -  Simulate non-bypassing loop extrusion by LEF with simulate()

- LEF_paths_gen.py - generate LEF paths on the genome several times and save into .h5 files