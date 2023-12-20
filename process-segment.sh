#!/bin/bash
set -e

SHARD="00000000"
SHARD_PREFIX=${SHARD:0:5}
LANGUAGE="zh"
WORLD_SIZE=2 # number of GPUs
USE_DEBUG="--debug"
WORK_DIR="/tmp/fish-speech"

echo "Cleaning up working directory"
rm -rf ${WORK_DIR}

echo "Downloading shard ${SHARD}"
mkdir -p ${WORK_DIR}/tars
huggingface-cli download fishaudio/playerfm audios/${SHARD_PREFIX}/${LANGUAGE}/${SHARD}.tar --local-dir ${WORK_DIR}/tars/
# cp /mnt/nas-hdd/datasets/playerfm/audios/${LANGUAGE}/${SHARD}.tar ${WORK_DIR}/tars
mkdir -p ${WORK_DIR}/audios

echo "Extracting shard ${SHARD}"
tar -xf ${WORK_DIR}/tars/${SHARD}.tar -C ${WORK_DIR}/audios/
rm -rf ${WORK_DIR}/tars

echo "Running VAD on shard ${SHARD}"
python fish_data_engine/tasks/segment_vad.py --input-dir ${WORK_DIR}/audios/ --output-dir ${WORK_DIR}/vads/ ${USE_DEBUG}
rm -rf ${WORK_DIR}/audios

echo "Running ASR on shard ${SHARD}"
python fish_data_engine/tasks/asr.py --input-dir ${WORK_DIR}/vads/ --output-dir ${WORK_DIR}/results/ ${USE_DEBUG}
rm -rf ${WORK_DIR}/vads

echo "Build tarball for shard ${SHARD}"
tar -cf ${WORK_DIR}/upload.tar -C ${WORK_DIR}/results/ .

echo "Uploading shard ${SHARD}"
huggingface-cli upload fishaudio/playerfm ${WORK_DIR}/upload.tar asr/${SHARD_PREFIX}/${LANGUAGE}/${SHARD}.tar

echo "Cleaning up working directory"
rm -rf ${WORK_DIR}
