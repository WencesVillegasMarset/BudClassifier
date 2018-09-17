
#!/bin/bash

## Configurar proxy
PROXY=""
#Descomentar para usar proxy
#PROXY="http_proxy='http://192.168.16.8:8080'"
PROGNAME=$(basename $0)

## Funcion de error
error_exit()
{

#   	----------------------------------------------------------------
#   	Function for exit due to fatal program error
#           	Accepts 1 argument:
#                   	string containing descriptive error message
#   	----------------------------------------------------------------


    	echo "${PROGNAME}: ${1:-"Unknown Error"}" 1>&2
    	exit 1
}

# Instalar dependencias
sudo $PROXY apt-get update
sudo $PROXY apt-get -y upgrade
sudo $PROXY apt-get -y remove x264 libx264-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install build-essential checkinstall cmake pkg-config yasm || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install git gfortran || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libjpeg-dev libpng-dev libtiff-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libavcodec-dev libavformat-dev libswscale-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libdc1394-22-dev libxine2-dev libv4l-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install ffmpeg qt5-default libgtk2.0-dev libtbb-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libatlas-base-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libfaac-dev libmp3lame-dev libtheora-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libvorbis-dev libxvidcore-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install x264 v4l-utils || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libprotobuf-dev protobuf-compiler || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libgoogle-glog-dev libgflags-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libopenexr-dev libtbb-dev libx264-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libqt4-dev libqt4-opengl-dev libdc1394-22-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libavcodec-dev libavformat-dev libswscale-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libvtk7-qt-dev || sudo $PROXY apt-get -y install libvtk6-qt-dev || sudo $PROXY apt-get -y install libvtk5-qt-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install default-jdk ant libboost-all-dev || error_exit "$LINENO: An error has occurred."
sudo $PROXY apt-get -y install libsuitesparse-dev || error_exit "$LINENO: An error has occurred."

# Creo carpeta de instalacion
mkdir -p ~/opencv_install || error_exit "$LINENO: An error has occurred."
if [ ! -f ./opencv-3.2.0.tar.gz ]; then
	wget -O opencv-3.2.0.tar.gz -F -c https://github.com/opencv/opencv/archive/3.2.0.tar.gz || error_exit "$LINENO: An error has occurred."
fi

cp ./opencv-3.2.0.tar.gz ~/opencv_install/

if [ ! -f ./opencv_contrib-3.2.0.zip ]; then
wget -O opencv_contrib-3.2.0.zip -F -c https://github.com/opencv/opencv_contrib/archive/3.2.0.zip || error_exit "$LINENO: An error has occurred."
fi

cp ./opencv_contrib-3.2.0.zip ~/opencv_install/

if [ ! -f ./ceres-solver-1.14.0.tar.gz ]; then
wget -c http://ceres-solver.org/ceres-solver-1.14.0.tar.gz || error_exit "$LINENO: An error has occurred."
fi
cp ./ceres-solver-1.14.0.tar.gz ~/opencv_install/
cd ~/opencv_install || error_exit "$LINENO: An error has occurred."

# Dependencias modulo SFM
tar zxf ceres-solver-1.14.0.tar.gz || error_exit "$LINENO: An error has occurred."
cd ceres-solver-1.14.0 || error_exit "$LINENO: An error has occurred."
mkdir -p build || error_exit "$LINENO: An error has occurred."
cd build || error_exit "$LINENO: An error has occurred."
cmake .. || error_exit "$LINENO: An error has occurred."
make -j$(nproc) || error_exit "$LINENO: An error has occurred."
sudo make install || error_exit "$LINENO: An error has occurred."
cd ~/opencv_install || error_exit "$LINENO: An error has occurred."

# Instalacion OpenCV 3.2.0
tar -xvzf opencv-3.2.0.tar.gz || error_exit "$LINENO: An error has occurred."
unzip -o opencv_contrib-3.2.0.zip || error_exit "$LINENO: An error has occurred."

cd opencv-3.2.0 || error_exit "$LINENO: An error has occurred."
mkdir -p build || error_exit "$LINENO: An error has occurred."
cd build || error_exit "$LINENO: An error has occurred."
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.2.0/modules -D BUILD_EXAMPLES=OFF .. || error_exit "$LINENO: An error has occurred."
make -j$(nproc) || error_exit "$LINENO: An error has occurred."
sudo make install || error_exit "$LINENO: An error has occurred."
echo '/usr/local/lib' | sudo tee --append /etc/ld.so.conf.d/opencv.conf || error_exit "$LINENO: An error has occurred."
sudo ldconfig || error_exit "$LINENO: An error has occurred."
echo 'PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig' | sudo tee --append ~/.bashrc || error_exit "$LINENO: An error has occurred."
echo 'export PKG_CONFIG_PATH' | sudo tee --append ~/.bashrc || error_exit "$LINENO: An error has occurred."
source ~/.bashrc || error_exit "$LINENO: An error has occurred."
echo ""
echo "##########################################################################"
echo "############  Muy bien! OpenCV-3.2.0 instalado correctamente  ############"
echo "##########################################################################"
echo ""
