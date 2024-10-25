# Define base image. Set HOME to avoid Matplotlib warning about non-writable
# MPLCONFIGDIR on Neurite import when running as non-root user.
FROM tensorflow/tensorflow:2.17.0-gpu AS base
ENV FREESURFER_HOME=/freesurfer
ENV PYTHONUSERBASE="$FREESURFER_HOME/env"
ENV PATH="$FREESURFER_HOME:$PATH"
ENV HOME=/tmp


# Intermediate build stage. Install Python packages to user base for easy COPY.
FROM base AS copy

COPY --chmod=0775 mri_synthmorph $FREESURFER_HOME/
COPY --chmod=0664 synthmorph/*.py $FREESURFER_HOME/synthmorph/
COPY --chmod=0664 synthmorph.*.h5 $FREESURFER_HOME/models/

RUN apt-get update && apt-get install -y --no-install-recommends git
RUN python3 -m pip install -U pip
RUN python3 -m pip install --user \
    'numpy<2.0' \
    git+https://github.com/adalca/pystrum.git@ba35d4b357f54e5ed577cbd413076a07ef810a21 \
    git+https://github.com/adalca/neurite.git@9ae2f5cec2201eedbcc6929cecf852193cef7646 \
    git+https://github.com/freesurfer/surfa.git@041905fca717447780e0cc211197669e3218de2f \
    git+https://github.com/voxelmorph/voxelmorph.git@53d1b95fa734648c92fd8af4f3807b09cb56c342

WORKDIR /artifacts
RUN python3 -V >python.txt
RUN python3 -m pip freeze >requirements.txt
RUN mri_synthmorph -h >help.general.txt
RUN mri_synthmorph register -h >help.register.txt
RUN mri_synthmorph apply -h >help.apply.txt


# Export Python requirements for reference. Build artifacts will only exist in
# in the target stage `export`.
FROM scratch AS export
COPY --from=copy /artifacts/*.txt /


# Exclude Git and caches from final image to save space. Copy only once to
# avoid unnecessary container layers. Set working directory to /mnt for data
# exchange with the host without having to specify the full path.
FROM base
COPY --from=copy $FREESURFER_HOME $FREESURFER_HOME
WORKDIR /mnt
ENTRYPOINT ["mri_synthmorph"]
