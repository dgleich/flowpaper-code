for fn in cut-graph-init cut-graph-R cut-graph_source cut-graph_reduce cut-graph_sinkFI cut-graph_sinkLFI; do
  echo $fn
  inkscape --without-gui $fn.svg --export-eps $fn.eps
done

