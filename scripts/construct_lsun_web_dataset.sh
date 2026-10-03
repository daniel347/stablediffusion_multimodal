
source="/path/to/lsun/bedroom"
dest="/path/to/destination"

# Generate depth maps for the LSUN bedroom dataset using Depth-Anything-V2

mkdir $source/colour
mkdir $source/depth
mv $source/**/*.png $source/colour/.

cd Depth-Anything-V2
python metric_depth/run.py --img-path $source/colour  --outdir $source/depth --dataset --n-images 1000000 --clip 30


if [[ ! -e $dest ]]; then
	mkdir $dest
fi

# Convert to web dataset format

web_dataset_dest="/path/to/web_dataset"
mkdir $web_dataset_dest
mkdir $web_dataset_dest/train
mkdir $web_dataset_dest/validation

shard_size=10000

declare -i idx
declare -i shard_count
idx=0
shard_count=0

for file in $dir/colour/**/*.png; do
  echo "checking file $file"
  depth_file=${file/colour/depth}
  echo "looking for depth file $depth_file"
  if [[ -e $depth_file ]]; then
    cp $file $dest/$idx.colour.png
    cp $depth_file $dest/$idx.depth.png
    idx+=1
    if idx % shard_size == 0; then
      echo "shard $shard_count completed"
      dest_shard="$web_dataset_dest/train/train-$(printf "%05d" $shard_count).tar"
      tar -cf $dest_shard $dest/*.png
      rm $dest/*.png
      $shard_count+=1
    fi
  fi
done

# Save the last shard
echo "last shard completed"
dest_shard="$web_dataset_dest/train/train-$shard_count.tar"
tar -cf $dest_shard $dest/*.png
rm $dest/*.png

# Convert the last two shards to validation set
mv $web_dataset_dest/train/train-$(printf "%05d" $((shard_count-1))) $web_dataset_dest/validation/validation-00000.tar
mv $web_dataset_dest/train/train-$(printf "%05d" $shard_count).tar $web_dataset_dest/validation/validation-00001.tar
