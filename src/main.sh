#!/bin/sh
set -e
set -o pipefail

function parse_yaml {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @ | tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $1 |
        awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])(".")}
         printf("--%s%s%s=%s\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

args=$(parse_yaml $1)
tmp=${args#*\=}        # Remove everything up to and including first "
runfile=${tmp%%\-\-*}    # Remove the first -- and everything following it
# python_args=--${tmp#*\-\-} # Remove everything up to second "

clargs="$*"
# prefix=${clargs%%\-\-*}    # Remove the first -- and everything following it
# python_clargs=${clargs#"$prefix"}

# echo $runfile
# echo $python_args

# echo "Running..."
# echo $runfile -C $clargs $python_args
# eval $runfile -C $clargs $python_args

# python_args=($python_args)

# for kv in "${python_args[@]}"; do
#     echo $kv
#     kvarr=(${kv//=/ })
#     k=${kvarr[0]}
#     if [[ "$clargs" == *"$k"* ]]; then
#         echo "Manually setting $kv"
#     else 
#         clargs="$clargs $kv"
    # fi
# done

# echo $runfile
# echo $clargs

echo $runfile -C $clargs
eval $runfile -C $clargs