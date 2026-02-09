#! /bin/bash

installation_scripts="$(dirname "$0")/install-scripts"
. $installation_scripts/install_multiple_support.sh
. $environment_scripts/install_mmposewholebody.sh
. $environment_scripts/install_mediapipe.sh
. $environment_scripts/install_openpifpaf.sh
