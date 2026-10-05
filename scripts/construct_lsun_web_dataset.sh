shopt -s globstar nullglob caseglob

source="../../datasets/lsun_mini/"
dest="../../datasets/lsun_mini_temp"
web_dataset_dest="../../datasets/lsun_mini_web"
colour_ext="jpg"
depth_ext="png"

if [[ ! -e $dest ]]; then
	mkdir $dest
fi

# Generate depth maps for the LSUN bedroom dataset using Depth-Anything-V2

mkdir $dest/colour
mkdir $dest/depth
for file in $source/**/*.jpg; do
  echo $file
  mv $file $dest/colour/.
done

cd Depth-Anything-V2
# python metric_depth/run.py --img-path ../$dest/colour  --outdir ../$dest/depth --dataset --n-images 1000000 --clip 30 --ext jpg
cd ../

if [ $? -ne 0 ]; then
  exit 1
fi

# Convert to web dataset format

mkdir $web_dataset_dest
mkdir $web_dataset_dest/train
mkdir $web_dataset_dest/validation

shard_size=10000

declare -i idx
declare -i shard_count
idx=0
shard_count=0

tmp_folder=$dest/shard
echo $tmp_folder
mkdir $tmp_folder

for file in $dest/colour/**/*.$colour_ext; do
  echo "checking file $file"
  depth_file=${file/colour/depth}
  depth_file=${depth_file/$colour_ext/$depth_ext}
  echo "looking for depth file $depth_file"
  if [[ -e $depth_file ]]; then
    echo "found depth file"
    cp $file $tmp_folder/$idx.colour.$colour_ext
    cp $depth_file $tmp_folder/$idx.depth.$depth_ext
    idx+=1
    if [[ $((idx % shard_size)) -eq 0 ]]; then
      echo "shard $shard_count completed"
      dest_shard="$web_dataset_dest/train/train-$(printf "%05d" $shard_count).tar"
      echo $dest_shard
      
      cd $tmp_folder
      tar -cf "$OLDPWD/$dest_shard" *
      rm ./*
      cd "$OLDPWD"
      
      $shard_count+=1
    fi
  fi
done

# Save the last shard
echo "last shard completed"
dest_shard="$web_dataset_dest/train/train-$(printf "%05d" $shard_count).tar"

echo $tmp_folder
cd $tmp_folder
tar -cf "$OLDPWD/$dest_shard" * --sort name
rm ./*
echo $OLDPWD
cd "$OLDPWD"



# Convert the last two shards to validation set
mv $web_dataset_dest/train/train-$(printf "%05d" $(($shard_count-1))) $web_dataset_dest/validation/validation-00000.tar
mv $web_dataset_dest/train/train-$(printf "%05d" $shard_count).tar $web_dataset_dest/validation/validation-00001.tar
