#!/bin/bash

set -x
PREFIX=${PREFIX:-}
PROJECT=${PROJECT:?"undefined PROJECT"}
ZONE=${ZONE:?"undefined ZONE"}
DATA_DIR=${DATA_DIR:?"undefined DATA_DIR"}
TPU_TYPE=v3-8
TASK=${TASK:?"undefined TASK"}
BATCH_SIZE=${BATCH_SIZE:-65536}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
if [ "$SPM" == "" ] || [ "$OUT_DIR" == "" ]; then
    exit 1
fi

LOG_DIR=${LOG_DIR:-log}
LOGFILE="${LOG_DIR}/${PREFIX}_$(date +'%Y-%m-%d-%k:%M:%S').log"
mkdir -p "${LOG_DIR}"
echo "Log: ${LOGFILE}"

BASENAME="$(basename "${OUT_DIR}")"
LOCAL_OUT_DIR=${LOCAL_OUT_DIR:-~/data/t5/${BASENAME}}
mkdir -p "${LOCAL_OUT_DIR}"
echo "Local out dir: ${LOCAL_OUT_DIR}"

GIN_LR=${GIN_LR:-learning_rate_schedules/rsqrt_no_ramp_down.gin}

(
    gsutil cp "${SPM}" "${OUT_DIR}"/
    if [ "$PRE" == "1" ]; then
        if [ "${TPU}" == "" ]; then
            TPU="tpu-${TPU_TYPE}-pre-${PREFIX}-${USER}-${RANDOM}"
        fi
        SAVE_CHECKPOINTS_STEPS=${SAVE_CHECKPOINTS_STEPS:-2000}
    else
        if [ "${TPU}" == "" ]; then
            echo "Please specify TPU NAME"
            exit 1
        fi
        ctpu up --name="${TPU}" --project="${PROJECT}" --zone="${ZONE}" \
            --tpu-size=${TPU_TYPE} --tpu-only --noconf
        SAVE_CHECKPOINTS_STEPS=${SAVE_CHECKPOINTS_STEPS:-7200}
    fi
    FIRST=1

    while true; do
        if [ "$PRE" == "1" ]; then
            if [ "${FIRST}" == "1" ]; then
                FIRST=0
            else
                gcloud --quiet compute tpus delete "$TPU" --zone "$ZONE"
            fi
            ctpu up --name="${TPU}" --project="${PROJECT}" --zone="${ZONE}" \
                --tpu-size=${TPU_TYPE} --tpu-only --noconf --preemptible
        else
            gcloud compute tpus start "$TPU" --zone "$ZONE"
        fi

        python -m t5.models.mesh_transformer_main \
            --tpu="${TPU}" \
            --gcp_project="${PROJECT}" \
            --tpu_zone="${ZONE}" \
            --model_dir="${OUT_DIR}" \
            --gin_file="dataset.gin" \
            --gin_file="models/t5.1.1.base.gin" \
            --gin_file="objectives/span_3_15_u_u.gin" \
            --gin_file="${GIN_LR}" \
            --gin_param="MIXTURE_NAME = '${TASK}'" \
            --gin_param="utils.run.sequence_length = {'inputs': 1024, 'targets': 256}" \
            --gin_param="utils.run.batch_size = ('tokens_per_batch', ${BATCH_SIZE})" \
            --gin_param="run.train_steps = ${TRAIN_STEPS}" \
            --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
            --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_TYPE}'" \
            --t5_tfds_data_dir="${DATA_DIR}" \
            --module_import="scripts.jalan.jalan" \
            --module_import="scripts.mywiki40b.mywiki40b" \
            --module_import="scripts.task" \
            --gin_param="run.save_checkpoints_steps=${SAVE_CHECKPOINTS_STEPS}" \
            --gin_param="run.keep_checkpoint_max=5" \
            && break
        sleep 5
    done

    if [ "$PRE" == "1" ]; then
        gcloud --quiet compute tpus delete "$TPU" --zone "$ZONE"
    else
        gcloud compute tpus stop "$TPU" --zone "$ZONE"
    fi

    python scripts/dist.py -t "${SPM}" -i "${OUT_DIR}" -o "${LOCAL_OUT_DIR}"
    cd "${LOCAL_OUT_DIR}"
    rm "t5.pt.${BASENAME}"
    ln -s pt "t5.pt.${BASENAME}"
    zip -r "t5.pt.${BASENAME}".zip "t5.pt.${BASENAME}"
    rm "t5.tf.${BASENAME}"
    ln -s tf "t5.tf.${BASENAME}"
    zip -r "t5.tf.${BASENAME}".zip "t5.tf.${BASENAME}"

) \
    2>&1 | tee -a "${LOGFILE}"
