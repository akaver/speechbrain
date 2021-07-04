# Data Preparation
When done with everything written here, then run voxceleb_prepare-akaver.py

# VoxCeleb1 and VoxCeleb2 dataset

Download all of voxceleb1 and voxceleb2 audio and meta files.
Keep them in separate catalogs voxceleb1 and voxceleb2.

So initially you should have:

\voxceleb1
 vox1_meta.csv
 iden_split.txt
 list_test_all.txt
 list_test_all2.txt
 list_test_hard.txt
 list_test_hard2.txt
 veri_test.txt
 veri_test2.txt
 vox1_dev_wav_partaa
 vox1_dev_wav_partab
 vox1_dev_wav_partac
 vox1_dev_wav_partad
 vox1_test_wav.zip

Merge dev set files together 
~~~bash
cat vox1_dev* > vox1_dev_wav.zip 
~~~

Unzip audio files into dev and test subdirs.
~~~bash
unzip vox1_dev_wav.zip -d dev
unzip vox1_test_wav.zip -d test
~~~

And in voxceleb2:

\voxceleb2
 vox2_dev_aac_partaa
 vox2_dev_aac_partab
 vox2_dev_aac_partac
 vox2_dev_aac_partad
 vox2_dev_aac_partae
 vox2_dev_aac_partaf
 vox2_dev_aac_partag
 vox2_dev_aac_partah
 vox2_test_aac.zip
 vox2_meta.csv

Merge dev set files together 
~~~bash
cat vox2_dev_aac* > vox2_aac.zip 
~~~

Unzip audio files into dev and test subdirs.
~~~bash
unzip vox2_aac.zip -d dev
unzip vox2_test_aac.zip -d test
~~~

Convert all the aac files to wav.
Or use 
https://gitmemory.com/issue/pytorch/audio/104/493137979 
or 
https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830

~~~bash
#!/bin/bash

SOURCE_DIR="./aac"
TARGET_DIR="./wav"
export SOURCE_DIR
export TARGET_DIR

doone() {
    sourceFile="$1"
    if [[ "$(basename "${sourceFile}")" != ._* ]] ; then # Skip files starting with "._"
    tmpVar="${sourceFile%.*}.wav"
    targetFile="${tmpVar/$SOURCE_DIR/$TARGET_DIR}"
    targetFilePath=$(dirname "${targetFile}")
    mkdir -p "${targetFilePath}"
    if [ ! -f "$targetFile" ]; then # If the target file doesn't exist already
    echo "Input: $sourceFile"
    echo "Output: $targetFile"
    ffmpeg -i "$sourceFile" "$targetFile" < /dev/null
    fi
    fi
}

export -f doone

# Find all m4a files in the given SOURCE_DIR and iterate over them:
find "${SOURCE_DIR}" -type f \( -iname "*.m4a" \) -print0 |
  parallel -0 doone
~~~
Place this script into dev and test directory and execute. End result should be the exact same structure as before - just in wav directory and all audio files converted to wav format.
Everything else should stay the same - file names and dir names.





*** TODO ***

# SITW - The Speakers in the Wild dataset

Download SITW dataset into directory sitw and unpack.

~~~bash
tar -zxvf sitw_database.v4.tar.gz
~~~

Sitw dataset is in flac format, convert to wav
