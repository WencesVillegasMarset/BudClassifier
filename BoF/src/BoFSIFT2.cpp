// BoFSIFT2.cpp : Defines the entry point for the console application.

#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>


using namespace boost::filesystem;
using namespace cv;
using namespace std;
using namespace xfeatures2d;


/**
 * Calcula los keypoints sobre una imagen completa, y luego sobre otra
 * imagen extraida de la primera. Grafica los keypoints calculados en
 * ambas imagenes y muestra la cantidad de keypoints encontrados en
 * las dos imagenes, asi como tambien en la region de la imagen completa
 * a la que corresponde la segunda.
 */
void pruebaKeypoints() {

	//The SIFT feature extractor and descriptor
	cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

	//read the image
	Mat img1 = imread("../../data/images/debug_pert130K/0032-184-1049_remain100_added1602_nyrel95_dx-0.16_dy0.1_pyo16491_py16491_pny264130_radio154.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints1;
	//detect feature points
	detector->detect(img1, keypoints1);

	printf("keypoints1: %i\n", (int) keypoints1.size());
	drawKeypoints(img1, keypoints1, img1);
	imwrite("../1-kp.jpg", img1);

//	vector<KeyPoint> keypointsCut;
//
//	for (int i = 0; i < (int) keypoints1.size(); i++) {
//		float x = keypoints1[i].pt.x;
//		float y = keypoints1[i].pt.y;
//		if (x >= 2500 && x <= 3500 && y >= 1500 && y <= 2500)
//			keypointsCut.push_back(keypoints1[i]);
//	}
//
//	printf("keypointsCut: %i\n", (int) keypointsCut.size());
//
//	//read the image
//	Mat img2 = imread("../2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	//To store the keypoints that will be extracted by SIFT
//	vector<KeyPoint> keypoints2;
//	//detect feature points
//	detector->detect(img2, keypoints2);
//
//	printf("keypoints2: %i\n", (int) keypoints2.size());
//	drawKeypoints(img2, keypoints2, img2);
//	imwrite("../2-kp.jpg", img2);

}


/**
 * Calcula los SIFT descriptors de la imagen en 'filename' y los
 * almacena en la matriz 'descriptor'. Si 'saveDesc' es true
 * guarda los descriptores y sus coordenadas X,Y en un archivo CSV.
 *
 * @param filename
 * @param detector
 * @param descriptor
 * @param saveAsCSV
 * @param saveCoordKP
 * @param outputCSV
 */
void getImgSIFTDescriptors(string filename, Ptr<SIFT>& detector,
		Mat& descriptor, bool saveAsCSV = false, bool saveCoordKP = false) {

	//path to file
	path p(filename);
	string stemFile = p.stem().string();
	string outputCSV = p.parent_path().string();
	//to store the descriptors
	char * fileDesc = new char[200];
	//create the file name of an image
	sprintf(fileDesc, "%s/%s-SIFT.csv", outputCSV.c_str(), stemFile.c_str() );

	//Try to read the keypoints
	Ptr<ml::TrainData> dataCSV = ml::TrainData::loadFromCSV(fileDesc, 0);
	//If CSV file of keypoints exists, load it
	if (!dataCSV.empty()) {
		hconcat(dataCSV->getSamples(), dataCSV->getResponses(), descriptor);
//		printf("\tSIFT descriptors loaded: %s\n", fileDesc);
		return;
	}

	//read image
	Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	if (img.empty()) {
		printf("\t#ERROR: no se puede cargar la imagen '%s'\n",
				filename.c_str());
	} else {
		//To store the keypoints that will be extracted by SIFT
		vector<KeyPoint> keypoints;

		//detect feature points
		detector->detect(img, keypoints);
		//compute the descriptors for each keypoint
		detector->compute(img, keypoints, descriptor);

		//
		if (saveAsCSV) {
			if (saveCoordKP) {
				Mat coordKP(descriptor.rows, 2, descriptor.type());

				for (int i=0; i<coordKP.rows; i++) {
					coordKP.at<float>(i,0)=keypoints[i].pt.x;
					coordKP.at<float>(i,1)=keypoints[i].pt.y;
				}

				hconcat(coordKP, descriptor, descriptor);
			}

			//open the file to write the resultant descriptor
			boost::filesystem::ofstream fs1(fileDesc);
			//image's info
			fs1 << format(descriptor, Formatter::FMT_CSV) << endl;
			//release the file storage
			fs1.close();

//			printf("\tSIFT descriptors saved: %s\n", fileDesc);

		}

	}

}


/**
 * Calcula los SIFT descriptors del conjunto de imagenes en 'imgFiles'
 * y los almacena TODOS en la matriz 'imagesDescriptor'. Si 'saveDesc'
 * es 'true' guarda a disco todos los descriptores (sin hacer refereancia
 * al nombre de la imagen ni a sus coordenadas) de todas las imagenes
 * procesadas.
 *
 * @param imgFiles
 * @param detector
 * @param imagesDescriptor
 * @param saveAsCSV
 * @param fileDesc
 */
void getAllImgSIFTDescriptors(vector<string>& imgFiles, Ptr<SIFT>& detector,
		Mat& imagesDescriptor, bool saveAsCSV = false, string fileDesc = "allDescriptors.csv") {

	for (vector<string>::iterator it = imgFiles.begin(); it != imgFiles.end(); ++it) {

		string filename = *it;
		//To delete page breaks accidentally included
		filename.erase(remove(filename.begin(), filename.end(), '\n'), filename.end());

		if (!(is_regular_file(filename)) || !(extension(filename) == ".jpg")) {
			cout << "\tWARNING: " << filename << " no se cargo\n";
			continue;
		}

		Mat descriptor;

		getImgSIFTDescriptors(filename, detector, descriptor, true, false);

		if (descriptor.empty()) {
			printf("\t#WARNING: No se detectaron keypoints para '%s'\n",
					filename.c_str());
			continue;
		}

		//put the all feature descriptors in a single Mat object
		imagesDescriptor.push_back(descriptor);

	}

	if (saveAsCSV) {
		//open the file to write the resultant descriptor
		boost::filesystem::ofstream fs1(fileDesc.c_str());
		//image's info
		fs1 << format(imagesDescriptor, Formatter::FMT_CSV) << endl;
		//release the file storage
		fs1.close();
	}

}


/**
 *
 * @param dictionary
 * @param featuresUnclustered
 * @param outputDict
 * @param dictSize
 * @param saveAsCSV
 */
void buildBoFDictionary(Mat& dictionary, Mat& featuresUnclustered,
		string outputDict, int dictSize, bool saveAsCSV = false) {

	//to store the input file names
	char * filedic = new char[200];
	//create the file name of an image
	sprintf(filedic, "%sdict-s%i.yml", outputDict.c_str(), dictSize);

	if (featuresUnclustered.empty()) {
		printf("ERROR en %s: No se detectaron keypoints\n", filedic);
		return;
	}

	//Construct BOWKMeansTrainer
	//Define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	//retries number
	int retries = 1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictSize, tc, retries, flags);

	//cluster the feature vectors
	dictionary = bowTrainer.cluster(featuresUnclustered);

	//store the vocabulary
	FileStorage fs(filedic, FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	if (saveAsCSV) {
		//To store the descriptor file name
		char * filecsv = new char[200];
		//the descriptor file with the location.
		sprintf(filecsv, "%sdict-s%i.csv", outputDict.c_str(), dictSize);
		//open the file to write the resultant descriptor
		boost::filesystem::ofstream fs1(filecsv);
		//image's info
		fs1 << format(dictionary, Formatter::FMT_CSV) << endl;
		//release the file storage
		fs1.close();
	}

	printf("\tDictionario %s creado [%i keypoints]\n", filedic,
			featuresUnclustered.rows);

}

/**
 *
 * @param outputDict
 * @param dictSize
 * @param dictionary
 */
void readBoFDictionary(string outputDict, int dictSize, Mat& dictionary, bool verbose=true) {
	//prepare BOW descriptor extractor from the dictionary

	//to store the input file names
	char * filedic = new char[100];
	//create the file name of an image
	sprintf(filedic, "%sdict-s%i.yml", outputDict.c_str(), dictSize);
	FileStorage fs(filedic, FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();

	if (verbose)
		printf("\tDictionario %s cargado\n", filedic);
}

/**
 *
 * @param prior
 * @param dictionary
 * @param bowDescriptor
 * @param SIFTdescriptors
 */
void computePriorHistogram(float prior, Mat& dictionary, Mat& bowDescriptor, int nroKps) {
	//Cantidad de bins del histograma (BoF descriptor)
	int words = dictionary.rows;
	float p = prior / words;
	if (bowDescriptor.empty()) {
		//Como esta vacio, sea el prior que sea, los bins del histograma son ctes
		bowDescriptor = Mat(1, words, dictionary.type(), Scalar::all(p));
	} else {
		//Recupera el histograma absoluto (contador)
		bowDescriptor = bowDescriptor * nroKps;
		//Suma al nro de keypoints un numero segun el prior
		nroKps = nroKps + prior;
		//Matriz temporal para almacenar el prior en cada bin
		Mat temp = Mat(1, words, dictionary.type(), Scalar::all(p));
		//Introduce el prior al descripor existente
		bowDescriptor = (bowDescriptor + temp) / (float)nroKps;
	}
}

/**
 * Calcula los BoF descriptors para cada imagen en 'imgFiles' y los
 * almacena en archivos CSV separados en el directorio 'outputHist'.
 *
 * @param dictionary
 * @param imgFiles
 * @param imgClass
 * @param detector
 * @param outputHist
 * @param prior
 */
void computeImgBoFDescriptor(Mat& dictionary, vector<string>& imgFiles, vector<string>& imgClass,
		Ptr<SIFT>& detector, string outputHist, float prior = 1) {

	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create a brute force matcher
	//Ptr<DescriptorMatcher> matcher(new BFMatcher);
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(detector, matcher);

	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);
	int dictSize = dictionary.rows;

	//To store the descriptor file name
	char * filecsv = new char[200];
	//the descriptor file with the location.
	sprintf(filecsv, "%shist-s%i.csv", outputHist.c_str(), dictSize);
	if (exists(filecsv)) {
		printf("\tHistograms loaded [%s]\n", outputHist.c_str());
		return;
	}
	//open the file to write the resultant descriptor
	boost::filesystem::ofstream fs1(filecsv);

	//iterates over all images in imgFiles
	for (int i = 0; i < signed(imgFiles.size()); i++) {

		string filename = imgFiles[i];
		string imageClass = imgClass[i];

		//To delete page breaks accidentally included
		filename.erase(remove(filename.begin(), filename.end(), '\n'), filename.end());
		imageClass.erase(remove(imageClass.begin(), imageClass.end(), '\n'), imageClass.end());

		//To store the keypoints that will be extracted by SIFT
		Mat keypoints;
		//Get image keypoints
		getImgSIFTDescriptors(filename, detector, keypoints, true, false);

		//To store the BoW (or BoF) representation of the image
		Mat bowDescriptor;

		if (!keypoints.empty()) {
			//extract BoW (or BoF) descriptor from given image
			bowDE.compute(keypoints, bowDescriptor);
		}

		//Prior
		if (prior != 0)
			computePriorHistogram(prior, dictionary, bowDescriptor,	signed(keypoints.rows));

		//Add bowDescriptor to CSV file
		if (!(bowDescriptor.empty())) {
			//image's info
			fs1 << filename << "," << format(bowDescriptor, Formatter::FMT_CSV) << "," << imageClass << endl;
		} else {
			printf("ERROR en %s: no se pudo calcular el BoF descriptor\n", filename.c_str());
		}

	}

	//release the file storage
	fs1.close();

	printf("\tHistograms computed [%s]\n", outputHist.c_str());

}

/**
 * Calcula los BoF descriptors para los 'SIFTdescriptors'.
 *
 * @param bowDE
 * @param imgDesc
 * @param outputDir
 * @param prior
 */
void computeImgBoFDescriptor(Mat& dictionary, Mat& SIFTdescriptors, Ptr<SIFT>& detector, Mat& bowDescriptor, float prior=1) {

	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(detector, matcher);

	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	//extract BoW (or BoF) descriptor from given image
	if ( !SIFTdescriptors.empty() )
		bowDE.compute(SIFTdescriptors, bowDescriptor);

	//Prior
	if (prior != 0)
		computePriorHistogram(prior, dictionary, bowDescriptor,	SIFTdescriptors.rows);

	if ( bowDescriptor.empty() ) {
		cout << "WARNING: no se pudo calcular el BoF descriptor\n";
	}

}


/* MAIN
 *
 */
int main(int argc, char** argv) {

	// no_argument			0
	// required_argument	1
	// optional_argument	2
	const struct option longopts[] = {
			{ "exp", required_argument, 0, 'e' },
			{ "imgList", required_argument, 0, 'l' },
			{ "imgClass", required_argument, 0, 'c' },
			{ "dictSize", required_argument, 0, 's' },
			{ "outDict", required_argument, 0, 'd' },
			{ "outDir", required_argument, 0, 'o' },
			{ "outFile", required_argument, 0, 'f' },
			{ "tempFile", required_argument, 0, 't' },
			{ "prior", required_argument, 0, 'p' },
			{ "infExt", required_argument, 0, 'i' },
			{ 0, 0, 0, 0 },
	};

	string experiment;
	vector<string> imgFiles;
	vector<string> imgClass;
	vector<int> dictSize;
	vector<string> tempDS;
	string outputDict;
	string outputDir;
	string outputFile;
	string tempFile;
	string infoExtra;
	float prior;

	// Parsing Arguments
	int index;
	int iarg = 0;
	while (iarg != -1) {
		iarg = getopt_long(argc, argv, "e:l:c:s:d:o:f:t:p:i:", longopts, &index);

		switch (iarg) {
		case 'e':
			experiment = optarg;
			break;

		case 'l':
			{
			std::ifstream t(optarg);
			std::stringstream buffer;
			buffer << t.rdbuf();
			string im(buffer.str());
			boost::split(imgFiles, im, boost::is_any_of(","));
			}
			break;

		case 'c':
			{
			std::ifstream t(optarg);
			std::stringstream buffer;
			buffer << t.rdbuf();
			string ic(buffer.str());
			boost::split(imgClass, ic, boost::is_any_of(","));
			}
			break;

		case 's':
			boost::split(tempDS, optarg, boost::is_any_of(","));
			for (int i=0; i<signed(tempDS.size()); i++)
				dictSize.push_back(atoi(tempDS[i].c_str()));
			break;

		case 'd':
			outputDict = optarg;
			break;

		case 'o':
			outputDir = optarg;
			break;

		case 'f':
			outputFile = optarg;
			break;

		case 't':
			tempFile = optarg;
			break;

		case 'p':
			prior = atof(optarg);
			break;

		case 'i':
			infoExtra = optarg;
			break;

		case '?':
			fprintf(stderr, "Usage: %s [?]\n", argv[0]);
			exit(EXIT_FAILURE);
			break;

		}
	}


	/* ############################################################# */
	/* Run experiments */

	if (experiment == "computeAllKp") { //Compute all keypoints

		cout << "COMPUTE ALL KEYPOINTS\n";

		cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
		Mat descriptor;

		for (vector<string>::iterator it = imgFiles.begin(); it != imgFiles.end(); ++it) {
			string filename = *it;
			//To delete page breaks accidentally included
			filename.erase(remove(filename.begin(), filename.end(), '\n'), filename.end());
			getImgSIFTDescriptors(filename, detector, descriptor, true, false);
		}

	} else 	if (experiment == "trainStageBoF") { //TRAIN STAGE BoF: Build Dictionary and compute train-histograms

		cout << "TRAIN STAGE BoF: Build Dictionary and compute train-histograms\n";

		//To store all the descriptors that are extracted from all the images.
		Mat featuresUnclustered;
		//The SIFT feature extractor and descriptor
		cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

		cout << "\tGet image SIFT descriptors\n";

		//Get image descriptors in imgFiles
		getAllImgSIFTDescriptors(imgFiles, detector, featuresUnclustered);

		cout << "\tCompute Dictionaries and BoF descriptors\n";

		for (int i=0; i< signed(dictSize.size()); i++) {
			//To store the builded dictionary.
			Mat dictionary;
			//Build dictionary
			buildBoFDictionary(dictionary, featuresUnclustered, outputDict, dictSize[i]);
			//Compute BoF descriptors
			computeImgBoFDescriptor(dictionary, imgFiles, imgClass, detector, outputDir, prior);
		}

	} else if (experiment == "testStageBoF") { //TEST STAGE BoF: Load dictionaries and compute test-histograms

		cout << "TEST STAGE BoF: Load dictionaries and compute test-histograms\n\n";

		for (int i=0; i< signed(dictSize.size()); i++) {
			//To store the dictionary.
			Mat dictionary;
			//Read the stored dictionary
			readBoFDictionary(outputDict, dictSize[i], dictionary);
			//The SIFT feature extractor and descriptor
			cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
			//Compute BoF descriptors
			computeImgBoFDescriptor(dictionary, imgFiles, imgClass, detector, outputDir, prior);
		}

	} else if (experiment == "computeKP") { //Compute SIFT keypoints
		//The SIFT feature extractor and descriptor
		cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

		//Compute and store keypoints of all imgFiles
		for (vector<string>::iterator it = imgFiles.begin(); it != imgFiles.end(); ++it) {
			string filename = *it;
			//To delete page breaks accidentally included
			filename.erase(remove(filename.begin(), filename.end(), '\n'), filename.end());
			if (!(is_regular_file(filename)) || !(extension(filename) == ".jpg")) {
				cout << "\tWARNING: " << filename << " no se cargo\n";
				continue;
			}
			Mat SIFTdescriptors;
			getImgSIFTDescriptors(filename, detector, SIFTdescriptors, true, true);
		}

	} else if (experiment == "windowed") {

		//The SIFT feature extractor and descriptor
		cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

		//Read CSV with the patch's keypoints
		Mat SIFTdescriptors;
		string filecsv = outputDir + tempFile;
		boost::filesystem::ifstream file(filecsv.c_str(), ios::binary | ios::ate);
		if (file.tellg()>0) {
			Ptr<ml::TrainData> raw_data = ml::TrainData::loadFromCSV(filecsv, 0, -2, 0);
			SIFTdescriptors = raw_data->getSamples();
		}

		//To store the dictionary.
		Mat dictionary;
		//Read the stored dictionary
		readBoFDictionary(outputDict, dictSize[0], dictionary, false);
		//To store the BoW (or BoF) representation of the image
		Mat bowDescriptor;
		//Compute BoF descriptors
		computeImgBoFDescriptor(dictionary, SIFTdescriptors, detector, bowDescriptor, prior);

		string filename = imgFiles[0];
		string fileHistCSV = outputDir + outputFile;
		//open the file to write the resultant descriptor
		boost::filesystem::ofstream fs1(fileHistCSV.c_str(), std::ios_base::app);
		//Add bowDescriptor to CSV file
		if (!(bowDescriptor.empty())) {
			//image's info
			fs1 << filename << "," << infoExtra << "," << format(bowDescriptor, Formatter::FMT_CSV) << endl;
		} else {
			printf("ERROR en %s: no se pudo calcular el BoF descriptor\n", filename.c_str());
		}

		//release the file storage
		fs1.close();

	} else if (experiment == "debugging") {
		cout << "DEBUGGING\n";
//		cout << "imgFiles: " << imgFiles[0] << "\n";
//		cout << "dictSize: " << dictSize[0] << "\n";
//		cout << "outputDict: " << outputDict << "\n";
//		cout << "outputDir: " << outputDir << "\n";
//		cout << "infoExtra: " << infoExtra << "\n";
//		cout << "prior: " << prior << "\n";


//		cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
//		Mat descriptor;
//		for (vector<string>::iterator it = imgFiles.begin(); it != imgFiles.end(); ++it) {
//			string filename = *it;
//			//To delete page breaks accidentally included
//			filename.erase(remove(filename.begin(), filename.end(), '\n'), filename.end());
//			getImgSIFTDescriptors(filename, detector, descriptor, true, false);
//		}

		float progress = 0.0;
		while (progress < 1.0) {
		    int barWidth = 70;
		    std::cout << "[";
		    int pos = barWidth * progress;
		    for (int i = 0; i < barWidth; ++i) {
		        if (i < pos) std::cout << "=";
		        else if (i == pos) std::cout << ">";
		        else std::cout << " ";
		    }
		    std::cout << "] " << int(progress * 100.0) << " %\r";
		    std::cout.flush();

		    progress += 0.05; // for demonstration only
		}
		std::cout << std::endl;
	}

	exit(0);
}
