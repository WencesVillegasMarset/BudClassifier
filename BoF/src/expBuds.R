rm(list=ls())

options(error = recover)
require(compiler)
enableJIT(3)

library(MASS)
library(jpeg)

library(doMC) # Multicore
cores <- 8 # Number of cores
registerDoMC(cores) # register cores

source("svm-bud-detection.R")
#source("perturbation_process.R")

# Parámetros para expBudDetection()
generalSeed <<- NULL
splitTrain <<- NULL
isBalancedTrain <<- NULL
isBalancedTest <<- NULL
corpusBaseBud <<- NULL
corpusBaseNoBud <<- NULL
fileCorpusCSV <<- NULL
inputImg <<- NULL
outputExp <<- NULL
outputExpRunTime <<-NULL
outputDict <<- NULL
outputTrainHist <<- NULL
outputTestHist <<- NULL
outputSVM <<- NULL
dictSize <<-NULL
prior <<- NULL

iterations <<- NULL

#Parámetros para expBudPerturbations()
outputPertHist <<- NULL
idPertDataset <<- NULL

# Parámetros para expTelecopicWindowed()
generalSeed <<- NULL
inputImg <<- NULL
outputExp <<- NULL
outputTelescKp <<- NULL
outputTelescHist <<- NULL
outputTelescClass <<- NULL
experiment <<- NULL
computeKP <<- NULL
computeBoF <<- NULL
computeClassSVM <<- NULL
computeImgDist <<- NULL

win_params <<- NULL
imgWidth <<- NULL
imgHeight <<- NULL

outputCSVMClass <<- NULL
outputOCSVMClass <<- NULL
rescale <<- NULL

useRAMdisk <<- NULL
outputTelescHistRAMdisk <<- NULL


expBudDetection_2016.03.25_R <- function() {

  # Parámetros del experimento
  # generalSeed <<- 103
  # iterations <<- c(1:4)
  # generalSeed <<- 104
  # iterations <<- c(5:8)
  generalSeed <<- 104
  iterations <<- c(1:10)
  general.ratios <<- c(1)
  cw <<- 0

  splitTrain <<- 0.7
  isBalancedTrain <<- FALSE
  isBalancedTest <<- TRUE
  corpusBaseBud <<- TRUE
  corpusBaseNoBud <<- FALSE
  fileCorpusCSV <<- "../../data/images/corpus-26000/corpus-26000.csv"
  inputImg <<- "../../data/images/corpus-26000/"
  outputExp <<- "../../output/exp_2016-03-25_R/"
  outputExpRunTime <<- paste0(outputExp,"runtimes/")
  outputDict <<- paste(outputExp, "dicts/", sep="")
  outputTrainHist <<- paste(outputExp, "trainHist/", sep="")
  outputTestHist <<- paste(outputExp, "testHist/", sep="")
  outputSVM <<- paste(outputExp, "svm/", sep="")
  dictSize <<- c(12,25,50,100,200,350,600)
  prior <<- 1

  runtimeAll <- system.time(expBudDetection())

}

expBudDetection_2016.04.01 <- function() {

  # Parámetros del experimento
  # generalSeed <<- 103
  # iterations <<- c(1:4)
  # generalSeed <<- 104
  # iterations <<- c(5:8)
  generalSeed <<- 104
  iterations <<- c(1)
  general.ratios <<- c(1)
  cw <<- 0

  splitTrain <<- 0.7
  isBalancedTrain <<- TRUE
  isBalancedTest <<- TRUE
  corpusBaseBud <<- TRUE
  corpusBaseNoBud <<- FALSE
  fileCorpusCSV <<- "../../data/images/corpus-26000/corpus-26000.csv"
  inputImg <<- "../../data/images/corpus-26000/"
  outputExp <<- "../../output/exp_2016-04-01/"
  outputExpRunTime <<- paste0(outputExp,"runtimes/")
  outputDict <<- paste(outputExp, "dicts/", sep="")
  outputTrainHist <<- paste(outputExp, "trainHist/", sep="")
  outputTestHist <<- paste(outputExp, "testHist/", sep="")
  outputSVM <<- paste(outputExp, "svm/", sep="")
  #dictSize <<- c(12,25,50,100,200,350,600)
  dictSize <<- c(600)
  prior <<- 1

  runtimeAll <- system.time(expBudDetection())

}

expBudDetection <- function() {
	# Si no existen los directorios de salida los crea
	if (!file.exists(outputExp))
		dir.create(outputExp)
	if (!file.exists(outputExpRunTime))
		dir.create(outputExpRunTime)		
	if (!file.exists(outputDict))
		dir.create(outputDict)
	if (!file.exists(outputTestHist))
		dir.create(outputTestHist)
	if (!file.exists(outputTrainHist))
		dir.create(outputTrainHist)

	set.seed(generalSeed)

	# Procesa el corpus de imágenes según los parámetros del experimento
	corpusCSV <- read.csv(fileCorpusCSV, header=TRUE, sep=",", stringsAsFactors=FALSE)

	# Filtra imágenes que no estan OK
	if ( corpusBaseBud | corpusBaseNoBud ) {
		# Filtra imágenes de yema que no están OK (tipo 2 y 3)
		idxBudType <- corpusCSV$class
		if (corpusBaseBud)
			idxBudType <- idxBudType & corpusCSV$type==1

		# Filtra imágenes de no-yema que no están OK (tipo 2 y 3)
		idxNoBudType <- !corpusCSV$class
		if (corpusBaseNoBud)
			idxNoBudType <- idxNoBudType & corpusCSV$type==1

		corpusCSV <- corpusCSV[ idxBudType | idxNoBudType, ]
	}

	# Splitea el dataset en train y test
	idxSplit <- sample(1:nrow(corpusCSV), nrow(corpusCSV)*splitTrain)
	trainset <- corpusCSV[idxSplit,]
	testset <- corpusCSV[-idxSplit,]

	# balancea el trainset
	if (isBalancedTrain) {
		idxBud <- which(trainset$class)
		idxNoBud <- which(!trainset$class)
		idxNoBud <- sample(idxNoBud, length(idxBud))
		# Descarta imágenes no-yema para balancear el trainset
		trainset <- trainset[ c(idxBud,idxNoBud), ]
	}

	# balancea el testset
	if (isBalancedTest) {
		idxBud <- which(testset$class)
		idxNoBud <- which(!testset$class)
		idxNoBud <- sample(idxNoBud)

		idxTemp <- 1:length(idxBud)
		if (!isBalancedTrain) {
			# Agrega al trainset todas las imágenes no-yema descartadas
			trainset <- rbind(trainset, testset[idxNoBud[-idxTemp],])
		}

		# Descarta imágenes no-yema para balancear el testset
		testset <- testset[ c( idxBud, idxNoBud[idxTemp] ), ]
	}

	# Prepara las imágenes del trainset a procesar
	trainImgList <- paste(inputImg, trainset$imageName, sep="", collapse=",")
	fileTrainImgList <- paste0(outputExp, "trainImgList.txt")
	write.table(trainImgList, file=fileTrainImgList, quote=F, sep="", row.names=F,col.names=F)

	trainImgClassList <- paste(trainset$class, sep="", collapse=",")
	fileTrainImgClassList <- paste0(outputExp, "trainImgClassList.txt")
	write.table(trainImgClassList, file=fileTrainImgClassList, quote=F, sep="", row.names=F,col.names=F)

	# Prepara las imágenes del testset a procesar
	testImgList <- paste(inputImg, testset$imageName, sep="", collapse=",")
	fileTestImgList <- paste0(outputExp, "testImgList.txt")
	write.table(testImgList, file=fileTestImgList, quote=F, sep="", row.names=F,col.names=F)

	testImgClassList <- paste(testset$class, sep="", collapse=",")
	fileTestImgClassList <- paste0(outputExp, "testImgClassList.txt")
	write.table(testImgClassList, file=fileTestImgClassList, quote=F, sep="", row.names=F,col.names=F)

	########################################################
	# Calcula los keypoints para todo el dataset
	runtimeComputedKP <- system.time({
	fileList <- c(fileTrainImgList, fileTestImgList)
	experiment <- "computeAllKp"
	foreach(i=1:length(fileList) ) %dopar% {
		fl <- fileList[i]
		cdmString <- paste("../Debug/BoW --exp ", experiment, " --imgList '", fl, "'", sep="")
		try(system(cdmString))
	}
	})

	print(runtimeComputedKP)

    foreach(i=1:length(dictSize) ) %dopar% {
    	ds <- dictSize[i]
		########################################################
		# Entrena los diccionarios y calcula descriptores para el trainset
		experiment <- "trainStageBoF"
		runtimeTrainDesc <- system.time({
		cdmString <- paste("../Debug/BoW --exp ", experiment, " --imgList '", fileTrainImgList,
							"' --imgClass '", fileTrainImgClassList,
							"' --dictSize '", ds, "' --outDict '", outputDict,
							"' --outDir '", outputTrainHist, "' --prior ", prior, sep="")
		try(system(cdmString))
		})

		rtvar <- runtimeTrainDesc
		outRT <- paste0(	outputExpRunTime, "s_",ds,"_",experiment,
							"_user_",rtvar[1],
							"_system_",rtvar[2],
							"_elapsed_",rtvar[3],
							".runtime")
		write(rtvar,outRT)

		########################################################
		# Calcula descriptores para el testset
		experiment <- "testStageBoF"
		runtimeTestDesc <- system.time({
		cdmString <- paste("../Debug/BoW --exp ", experiment, " --imgList '", fileTestImgList,
							"' --imgClass '", fileTestImgClassList,
							"' --dictSize '", ds, "' --outDict '", outputDict,
							"' --outDir '", outputTestHist, "' --prior ", prior, sep="")
		try(system(cdmString))
		})

		rtvar <- runtimeTestDesc
		outRT <- paste0(	outputExpRunTime, "s_",ds,"_",experiment,
							"_user_",rtvar[1],
							"_system_",rtvar[2],
							"_elapsed_",rtvar[3],
							".runtime")
		write(rtvar,outRT)

	}

	#######################################################

	 for (r in general.ratios) {
	 	#Entrena los modelos SVM y calcula las medidas de rendimiento sobre el trainset y testset
	 	for (ds in dictSize) {
	 		for (i in iterations) {
	 			set.directories(outputTrainHist, outputTestHist, outputSVM, outputExpRunTime)
	 			run.bdml.experiment(dic_pattern=ds, ratio=r, it=i, cw=cw)
	 		}
	 	}
	 }
}


#######################################################################
#######################################################################

expBudPerturbations_2015.03.03_BakGnd3000 <- function() {

	# Parámetros del experimento
	generalSeed <<- 103
	fileCorpusCSV <<- "../../data/images/TestCorpusBakGnd3000/output.csv"
	inputImg <<- "../../data/images/TestCorpusBakGnd3000/"
	outputExp <<- "../../output/exp_2016-03-25_R/"
	outputDict <<- paste(outputExp, "dicts/", sep="")
	outputPertHist <<- paste(outputExp, "TestCorpusBakGnd3000/", sep="")
	outputSVM <<- paste(outputExp, "svm/", sep="")
	prior <<- 1
  	dictSize <<- c(12,25,50,100,200,350,600)

  	iterations <- c(1:10)
	general.ratios <- c(1)
	
	for (r in general.ratios) {
		for (i in iterations) {
			expBudPerturbations(ratio=r, it=i)
		}
	}

}

expBudPerturbations_2016.03.25_R_win130k <- function() {

	# Parámetros del experimento
	generalSeed <<- 103
	fileCorpusCSV <<- "../../data/images/perturbWindows130K/perturbInfo.csv"
	inputImg <<- "../../data/images/perturbWindows130K/"
	outputExp <<- "../../output/exp_2016-03-25_R/"
	outputDict <<- paste(outputExp, "dicts/", sep="")
	outputPertHist <<- paste(outputExp, "pertHistWin130k/", sep="")
	outputSVM <<- paste(outputExp, "svm/", sep="")
	prior <<- 1
  	dictSize <<- c(12,25,50,100,200,350,600)

  	iterations <- c(1:10)
	general.ratios <- c(1)
	
	for (r in general.ratios) {
		expBudPerturbations(ratio=r, iterations=iterations)
	}

	process_2016.03.25_R_win130k()
}


expBudPerturbations <- function(ratio=-1, iterations=0) {
	# Si no existen los directorios de salida los crea
	if (!file.exists(outputExp))
		dir.create(outputExp)
	if (!file.exists(outputDict))
		dir.create(outputDict)
	if (!file.exists(outputPertHist))
		dir.create(outputPertHist)

	set.seed(generalSeed)

	# Procesa el corpus de imágenes según los parámetros del experimento
	perturbationSet <- read.csv(fileCorpusCSV, header=TRUE, sep=",", stringsAsFactors=FALSE)

	# Prepara las imágenes del dataset perturbado a procesar
	pertImgList <- paste(inputImg, perturbationSet$filename, sep="", collapse=",")
	filePertImgList <- paste0(outputExp, "pertImgList.txt")
	write.table(pertImgList, file=filePertImgList, quote=F, sep="", row.names=F,col.names=F)

	# pertImgClassList <- paste( rep(TRUE,nrow(perturbationSet)), sep="", collapse=",")
	pertImgClassList <- paste( rep(FALSE,nrow(perturbationSet)), sep="", collapse=",")
	filePertImgClassList <- paste0(outputExp, "pertImgClassList.txt")
	write.table(pertImgClassList, file=filePertImgClassList, quote=F, sep="", row.names=F,col.names=F)

	# Clasifica el perturbationSet
	runtimePertPred <- system.time({

		dsstr <- paste0(dictSize,collapse=",")
		# Calcula descriptores para el perturbationSet
		experiment <- "testStageBoF"
		cdmString <- paste("../Debug/BoW --exp ", experiment, " --imgList '", filePertImgList,
								"' --imgClass '", filePertImgClassList,
								"' --dictSize '", dsstr, "' --outDict '", outputDict,
								"' --outDir '", outputPertHist, "' --prior ", prior, sep="")
		try(system(cdmString))

	    #for (ds in dictSize) {           # singlecore
		foreach(i=1:length(dictSize)) %dopar% { 
		  ds <- dictSize[i]

		  	for (it in iterations) {
			  	cat("\n\t## Dict:", ds, "\t\t## it:", it,"\n")
			    resultsPert <- NULL

			    if (ratio==-1)
			      	filename <- paste0(outputSVM,"s",ds,"_csvm_bestmodel.RData")
			    else
			      	filename <- paste0(outputSVM,"s",ds,"_r",ratio,"_it",it,"_csvm_bestmodel.RData")

			    if (file.exists(filename)){
			        load(filename)

			        pertCSVFile <- paste0(outputPertHist,"hist-s",ds,".csv")
			        pertData <- read.csv(pertCSVFile, header=FALSE, sep=",", stringsAsFactors=FALSE)

			        resultsPert <- predict(c_svm_bestmodel,
			                                 newdata=pertData,
			                                 probability=TRUE)
			        resultsPertProb <- attr(resultsPert, "probabilities")
			        resultsPert <- as.matrix(resultsPert)
			        resultsPert <- factor(resultsPert,levels=c("YEMA","NOYEMA"))

			        predFile <- NULL
			       	if (ratio==-1)
			      		predFile <- paste0(outputSVM,"s",ds,"_csvm_predict_pert.csv")
			      	else
			        	predFile <- paste(outputPertHist,"s",ds,"_r",ratio,"_it",it,"_csvm_predict_pert.csv",sep="")
			        
			        x <- cbind(imageName=pertData[,1], pred=as.character(resultsPert),
			                   probTrue=resultsPertProb[,1],
			                   probFalse=resultsPertProb[,2])
			        cat(paste("Saving:",predFile,"\n"))
			        write.table(x,file=predFile,sep=",", row.names=FALSE)

			    }else{
			      	cat(paste("WARNING: No existe ", filename ,"\n"))
			    }
	  	  	}
	    }
	})

}


# expTelescopicWindowed_2015.10.28 <- function() {

# 	# Parámetros del experimento
# 	generalSeed <<- 1
# 	inputImg <<- "../../data/images/telescopic/"
# 	outputExp <<- "../../output/exp_2015-10-28/"
# 	outputTelescKp <<- "../../output/telescKp/"
# 	outputTelescHist <<- paste(outputExp, "telescHist/", sep="")
# 	outputTelescClass <<- paste(outputExp, "telescClass/", sep="")
# 	outputDict <<- paste(outputExp, "dicts/", sep="")
# 	outputSVM <<- paste(outputExp, "svm/", sep="")
# 	dictSize <<- c("100","150","200","250","300","350","400","450","500","550","600","650")
# 	prior <<- 1

# 	# Para el cálculo de las windowed
# 	# Parámetros de los patchs y de las imágenes
# 	# level: 	1		2		3
# 	# MEAN: 	584,5	163,6	62,3
# 	# SD:		52,5	61,2	16,2
# 	win_params <<- list(level=c("01","02","03"), size=c(600,180,75), step=c(200,60,25))
# 	imgWidth <<- 5472
# 	imgHeight <<- 3648

# 	# Para el computo de la distribución de patches detectados
# 	outputCSVMClass <<- paste(outputTelescClass, "plot-c-svm/", sep="")
# 	outputOCSVMClass <<- paste(outputTelescClass, "plot-oc-svm/", sep="")
# 	rescale <<- 0.4

# 	# Solo se usa en el computo de los BoF histograms (computeBoF==TRUE)
# 	useRAMdisk <<- TRUE
# 	outputTelescHistRAMdisk <<- "../../output/tempRAMdisk/"

# 	computeKP <<- FALSE
# 	computeBoF <<- FALSE
# 	computeClassSVM <<-FALSE
# 	computeImgDist <<- FALSE

# 	expTelescopicWindowed()

# }

# expTelescopicWindowed_2015.10.30 <- function() {

# 	# Parámetros del experimento
# 	generalSeed <<- 1
# 	inputImg <<- "../../data/images/telescopic/"
# 	outputExp <<- "../../output/exp_2015-10-30/"
# 	outputTelescKp <<- "../../output/telescKp/"
# 	outputTelescHist <<- paste(outputExp, "telescHist/", sep="")
# 	outputTelescClass <<- paste(outputExp, "telescClass/", sep="")
# 	outputDict <<- paste(outputExp, "dicts/", sep="")
# 	outputSVM <<- paste(outputExp, "svm/", sep="")
# 	dictSize <<- c("100","150","200","250","300","350","400","450","500","550","600","650")
# 	prior <<- 1

# 	# Para el cálculo de las windowed
# 	# Parámetros de los patchs y de las imágenes
# 	# level: 	1		2		3
# 	# MEAN: 	584,5	163,6	62,3
# 	# SD:		52,5	61,2	16,2
# 	win_params <<- list(level=c("01","02","03"), size=c(600,180,75), step=c(200,60,25))
# 	imgWidth <<- 5472
# 	imgHeight <<- 3648

# 	# Para el computo de la distribución de patches detectados
# 	outputCSVMClass <<- paste(outputTelescClass, "plot-c-svm/", sep="")
# 	outputOCSVMClass <<- paste(outputTelescClass, "plot-oc-svm/", sep="")
# 	rescale <<- 0.4

# 	# Solo se usa en el computo de los BoF histograms (computeBoF==TRUE)
# 	useRAMdisk <<- TRUE
# 	outputTelescHistRAMdisk <<- "../../output/tempRAMdisk/"

# 	computeKP <<- FALSE
# 	computeBoF <<- TRUE
# 	computeClassSVM <<- TRUE
# 	computeImgDist <<- TRUE

# 	expTelescopicWindowed()

# }

# expTelescopicWindowed_2015.11.07 <- function() {

# 	# Parámetros del experimento
# 	generalSeed <<- 1
# 	inputImg <<- "../../data/images/telescopic/"
# 	outputExp <<- "../../output/exp_2015-11-07/"
# 	outputTelescKp <<- "../../output/telescKp/"
# 	outputTelescHist <<- paste(outputExp, "telescHist/", sep="")
# 	outputTelescClass <<- paste(outputExp, "telescClass/", sep="")
# 	outputDict <<- paste(outputExp, "dicts/", sep="")
# 	outputSVM <<- paste(outputExp, "svm/", sep="")
# 	dictSize <<- c("100","150","200","250","300","350","400","450","500","550","600","650")
# 	prior <<- 1

# 	# Para el cálculo de las windowed
# 	# Parámetros de los patchs y de las imágenes
# 	# level: 	1		2		3
# 	# MEAN: 	584,5	163,6	62,3
# 	# SD:		52,5	61,2	16,2
# 	win_params <<- list(level=c("01","02","03"), size=c(1200,360,150), step=c(400,120,50))
# 	imgWidth <<- 5472
# 	imgHeight <<- 3648

# 	# Para el computo de la distribución de patches detectados
# 	outputCSVMClass <<- paste(outputTelescClass, "plot-c-svm/", sep="")
# 	outputOCSVMClass <<- paste(outputTelescClass, "plot-oc-svm/", sep="")
# 	rescale <<- 0.4

# 	# Solo se usa en el computo de los BoF histograms (computeBoF==TRUE)
# 	useRAMdisk <<- TRUE
# 	outputTelescHistRAMdisk <<- "../../output/tempRAMdisk/"

# 	computeKP <<- FALSE
# 	computeBoF <<- FALSE
# 	computeClassSVM <<- TRUE
# 	computeImgDist <<- TRUE

# 	expTelescopicWindowed()

# }


# expTelescopicWindowed <- function() {

# 	# Si no existen los directorios de salida los crea
# 	if (!file.exists(outputExp))
# 		dir.create(outputExp)
# 	if (!file.exists(outputTelescHist))
# 		dir.create(outputTelescHist)

# 	###############################################
# 	# Lee las imágenes y le calcula sus keypoints #
# 	if (computeKP) {
# 		if (!file.exists(outputTelescKp))
# 			dir.create(outputTelescKp)

# 		imgTeles <- list.files(inputImg, "*.jpg")
# 		imgList <- paste(inputImg, imgTeles, sep="", collapse=",")

# 		experiment <- "computeKP"
# 		cdmString <- paste("../Debug/BoW --exp '", experiment, "' --imgList '", imgList,
# 							"' --outDir '", outputTelescKp, "'", sep="")
# 		try(system(cdmString))
# 	}

# 	#############################################################################
# 	# Calcula los BoF descriptors para todos los patches de todas las imágenes #
# 	if (computeBoF) {

# 		tempOutTelescHist <- NULL

# 		# Si vamos a usar RAM disk, crea el directorio (si no existe) y monta el disco en RAM
# 		if (useRAMdisk) {
# 			if (!file.exists(outputTelescHistRAMdisk))
# 				dir.create(outputTelescHistRAMdisk)
# 			cdmString <- paste0("echo ****** | sudo -S mount tmpfs ", outputTelescHistRAMdisk," -t tmpfs -o size=2G")
# 			try(system(cdmString))

# 			tempOutTelescHist <- outputTelescHist
# 			outputTelescHist <- outputTelescHistRAMdisk
# 		}

# 		experiment <<- "windowed"

# 		# Lee los archivos CSV con los KP de cada imagen telescopica
# 		csvTeles <- list.files(outputTelescKp, "*.csv")

# 		# Para cada diccionario en outputDict calcula los patch de cada imagen
#   		# for (ds in dictSize) { 						#singelcore
#   		foreach(i=1:length(dictSize) ) %dopar% {  	# multicore
#   			ds <- dictSize[i]						# Esta 2 lineas se comentan/descomentan juntas

# 			tmpf <- paste("temp-",ds,".csv", sep="")
# 			fileTemp <- paste(outputTelescHist, tmpf, sep="")

# 			for (csvFile in csvTeles) {

# 				a <- system.time({
# 				# Parámetros de los patches
# 				nroTelesc <- strsplit(csvFile, "-")[[1]][1]
# 				levelTelesc <- strsplit(csvFile, "-")[[1]][2]
# 				idxLevel <- levelTelesc == win_params$level
# 				sizeWin <- win_params$size[idxLevel]
# 				stepWin <- win_params$step[idxLevel]

# 				# Crea un CSV por cada diccionario y cada archivo, en el que se van almacenando los descriptores de cada patch
# 				fileHistBase <- paste("hist-s",ds,"-",nroTelesc,"-",levelTelesc,".csv", sep="")
# 				fileHistCSV <- paste(outputTelescHist, fileHistBase, sep="")
# 				headerCSV <- paste("imageName", "nroTelesc", "levelTelesc", "rowPatch", "colPatch", "nroPatch", paste(1:ds, collapse=","), sep=",")
# 				write.table(headerCSV, fileHistCSV, sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)

# 				csvFile <- paste(outputTelescKp, csvFile, sep="")
# 				# imgKP[,c(1,2)] coordenadas del keypoint; imgKP[,3:130] SIFT descriptor
# 				imgKP <- read.csv(csvFile, header=FALSE, sep=",", stringsAsFactors=FALSE)
# 				coordKP <- as.matrix(imgKP[,c(1,2)])
# 				descKP <- as.matrix(imgKP[,-c(1,2)])

# 				# Itera para cada patch de cada imagen (en este caso de cada CSV de los KP de la imagen)
# 				for ( colPatch in seq(0, (imgWidth-sizeWin), stepWin) ) {
# 					idxX <- coordKP[,1]>=colPatch & coordKP[,1]<(colPatch+sizeWin)

# 					for ( rowPatch in seq(0, (imgHeight-sizeWin), stepWin) ) {
# 						idxY <- coordKP[,2]>=rowPatch & coordKP[,2]<(rowPatch+sizeWin)
# 						# Extrae los keypoints que corresponde al patch
# 						patch <- descKP[idxX&idxY,]
# 						if (class(patch)=="integer") patch <- t(patch)

# 						# Guarda los keypoints del patch en un archivo temporal (se sobreescribe en cada iteración)
# 						write.table(patch, fileTemp, sep=",", row.names=FALSE, col.names=FALSE)

# 						# Prepara info para enviar al codigo BoF (c++)
# 						rp <- rowPatch/stepWin
# 						cp <- colPatch/stepWin
# 						nroPatch <- cp*trunc(1+(imgHeight-sizeWin)/stepWin) + rp + 1
# 						infoExtra <- paste(nroTelesc, levelTelesc, rp, cp, nroPatch,sep=",")

# 						cdmString <- paste("../Debug/BoW --exp ", experiment, " --imgList '", csvFile,"'",
# 						" --dictSize ", ds," --outDict '", outputDict,"' --outDir '", outputTelescHist,"'",
# 						" --outFile '", fileHistBase, "' --tempFile '", tmpf, "'",
# 						" --infExt '", infoExtra,"' --prior ", prior, sep="")

# 						try(system(cdmString))

# 					}

# 				}

# 				# Si usamos RAM disk, mueve todos los archivos al directorio permanente
# 				if (useRAMdisk) {
# 					to <- paste0(tempOutTelescHist, fileHistBase)
# 					file.copy(fileHistCSV, to)
# 					file.remove(fileHistCSV)
# 				}})

# 				cat("File: ", csvFile, " processed\n")
# 				print(a)

# 			}

# 			file.remove(fileTemp)
# 		}

# 		# Si usamos RAM disk, desmonta el RAM disk y elimina su directorio
# 		if (useRAMdisk) {
# 			cdmString <- paste0("echo reygarufa | sudo -S umount ", outputTelescHistRAMdisk)
# 			try(system(cdmString))

# 			file.remove(outputTelescHistRAMdisk)

# 			outputTelescHist <- tempOutTelescHist
# 		}

# 	}

# 	###################################################################
# 	# Clasifica cada patch de cada imagen con los mejores modelos SVM #
# 	if (computeClassSVM) {
# 		if (!file.exists(outputTelescClass))
# 			dir.create(outputTelescClass)

# 		bofFiles <- list.files(outputTelescHist, "*.csv")

# 		# for ( bf in bofFiles ) {					# singlecore
#   		foreach(i=1:length(bofFiles) ) %dopar% {  	# multicore
#   			bf <- bofFiles[i]						# Esta 2 lineas se comentan/descomentan juntas

#   			bofDataset <- read.csv(paste(outputTelescHist, bf, sep=""), header=TRUE,
#   									sep=",", stringsAsFactors=FALSE)
#   			classDataset <- bofDataset[,1:6]
#   			bofDataset <- bofDataset[,-c(1:6)]
#   			colnames(bofDataset) <- NULL

#   			prefijo <- unlist(strsplit(bf, "-"))[2]

# 			# C-SVM (two-class)
# 			bestModelCSVM <- paste(outputSVM, prefijo, "_csvm_bestmodel.RData", sep="")
# 			results.c_svm <- NA
# 			if (file.exists(bestModelCSVM)){
# 				# c_svm_bestmodel se carga desde bestModelCSVM (.RData file)
# 				load(bestModelCSVM)
# 				results.c_svm <- predict(c_svm_bestmodel,
# 									newdata=bofDataset,
# 									probability = TRUE)
# 				results.c_svm.prob <- attr(results.c_svm, "probabilities")
# 				results.c_svm <- as.matrix(results.c_svm)
# 				results.c_svm <- factor(results.c_svm,levels=c("TRUE","FALSE"))
# 			}else{
# 				cat("WARNING: no existe el archivo ", bestModelCSVM, "\n")
# 			}

# 			# OC-SVM (one-class)
# 			# bestModelOCSVM <- paste(outputSVM, prefijo, "_ocsvm_bestmodel.RData", sep="")
# 			# results.oc_svm <- NA
# 			# if (file.exists(bestModelOCSVM)){
# 			# 	# oc_svm_bestmodel se carga desde bestModelOCSVM (.RData file)
# 			# 	load(bestModelOCSVM)
# 			# 	results.oc_svm <- predict(oc_svm_bestmodel,
# 			# 						newdata=bofDataset)
# 			# 	results.oc_svm <- as.matrix(results.oc_svm)
# 			# 	results.oc_svm <- factor(results.oc_svm,levels=c("TRUE","FALSE"))
# 			# }else{
# 			# 	cat("WARNING: no existe el archivo ", bestModelOCSVM, "\n")
# 			# }

# 			classDataset <- cbind(classDataset, csvmClass=results.c_svm, csvmProbTRUE=results.c_svm.prob[,1])
# 			# classDataset <- cbind(classDataset, csvmClass=results.c_svm,
# 			# 						csvmProbTRUE=results.c_svm.prob[,1], ocsvmClass=results.oc_svm)
# 			bf <- paste("class", substring(bf, 5), sep="")
# 			classFile <- paste(outputTelescClass, bf, sep="")
# 			write.table(classDataset, classFile, sep=",", row.names=FALSE, col.names=TRUE, quote=FALSE)

# 			cat("Saved file ", classFile, "\n")

# 			rm(c_svm_bestmodel)
# 			# rm(oc_svm_bestmodel)
# 		}
# 	}


# 	######################################################################################
# 	# Calcula la distribución de los patches clasificados como yema y lo guarda como jpg #
# 	if (computeClassSVM) {
# 		if (!file.exists(outputCSVMClass))
# 			dir.create(outputCSVMClass)
# 		if (!file.exists(outputOCSVMClass))
# 			dir.create(outputOCSVMClass)

# 		plot.distribution()
# 	}

# }

# plot.distribution <- function() {

# 	# Lee el archivo con la info de las imagenes telescopicas
# 	infoTelesc <- read.csv(paste(inputImg,"telescopic.csv",sep=""), header=T)

# 	telescImages <- list.files(inputImg, "*.jpg")

# 	# for (img in telescImages) {						# singlecore
# 	foreach(i=1:length(telescImages) ) %dopar% {  	# multicore
# 		img <- telescImages[i]						# Esta 2 lineas se comentan/descomentan juntas

# 		imgName <- unlist(strsplit(img, ".jpg"))
# 		idxLevel <- unlist(strsplit(imgName, "-"))[2]==win_params$level
# 		sizeWin <- win_params$size[idxLevel]
# 		stepWin <- win_params$step[idxLevel]

# 		##### INFO TELESCOPIC IMAGES
# 		idxInfImg <- infoTelesc$originImg==img
# 		infoImage <- infoTelesc[idxInfImg,]

# 		xBud <- infoImage$xBudCenter
# 		yBud <- imgHeight-infoImage$yBudCenter

# 		for (ds in dictSize) {

# 			##### INFO PATCHES C-SVM
# 			fileout <- paste(outputCSVMClass, imgName, "-s", ds, ".jpg", sep="")
# 			if (!file.exists(fileout)) {

# 				filePatches <- paste(outputTelescClass, "class-s", ds, "-", imgName, ".csv", sep="")
# 				infoPatches <- read.csv(filePatches, header=T)
# 				idxClassTrue <- infoPatches$csvmClass
# 				infoPatches <- infoPatches[idxClassTrue,]

# 				x <- infoPatches$colPatch*stepWin + sizeWin/2
# 				y <- infoPatches$rowPatch*stepWin + sizeWin/2

# 				y <- imgHeight-y

# 				# PLOTS
# 				jpeg(fileout, width=imgWidth*rescale , height=imgHeight*rescale )

# 				par(mar=c(0,0,0,0))
# 				plot(c(0, imgWidth), c(0, imgHeight), type = "n", xlab = "", ylab = "", axes=FALSE)

# 				pathImg <- paste(inputImg,imgName,".jpg", sep="")
# 				anImage <- readJPEG(pathImg)
# 				rasterImage(anImage, 0,0,imgWidth,imgHeight, interpolate = FALSE)

# 				try({
# 					bivn.kde <- kde2d(x, y, n=2000*rescale, lims=c(0,imgWidth,0,imgHeight))
# 					image(bivn.kde, col=gray((1:10),alpha = 0.5), add=TRUE)
# 				})
# 				points(x=x, y=y, pch=19, cex=.7, col="red")
# 				points(x=xBud, y=yBud, pch=19, cex=1.5, col="blue")

# 				dev.off()
# 				cat("Saved file ", fileout, "\n")
# 			} else {
# 				cat("File ", fileout, " already exists\n")
# 			}

			# ##### INFO PATCHES OC-SVM
			# fileout <- paste(outputOCSVMClass, imgName, "-s", ds, ".jpg", sep="")
			# if (!file.exists(fileout)) {
			# 	filePatches <- paste(outputTelescClass, "class-s", ds, "-", imgName, ".csv", sep="")
			# 	infoPatches <- read.csv(filePatches, header=T)
			# 	idxClassTrue <- infoPatches$ocsvmClass
			# 	infoPatches <- infoPatches[idxClassTrue,]

			# 	x <- infoPatches$colPatch*stepWin + sizeWin/2
			# 	y <- infoPatches$rowPatch*stepWin + sizeWin/2

			# 	y <- imgHeight-y

			# 	# PLOTS
			# 	jpeg(fileout, width=imgWidth*rescale , height=imgHeight*rescale )

			# 	par(mar=c(0,0,0,0))
			# 	plot(c(0, imgWidth), c(0, imgHeight), type = "n", xlab = "", ylab = "", axes=FALSE)

			# 	pathImg <- paste(inputImg,imgName,".jpg", sep="")
			# 	anImage <- readJPEG(pathImg)
			# 	rasterImage(anImage, 0,0,imgWidth,imgHeight, interpolate = FALSE)

			# 	try({
			# 		bivn.kde <- kde2d(x, y, n=2000*rescale, lims=c(0,imgWidth,0,imgHeight))
			# 		image(bivn.kde, col=gray(c(0,.25,.5,.75,1),alpha = 0.5), add=TRUE)
			# 	})
			# 	points(x=x, y=y, pch=19, cex=.7, col="red")
			# 	points(x=xBud, y=yBud, pch=19, cex=1.5, col="blue")

			# 	dev.off()
			# 	cat("Saved file ", fileout, "\n")
			# } else {
			# 	cat("File ", fileout, " already exists\n")
			# }
# 
# 		}
# 	}
# }


# expLC1 <- function() {
# 	# Parametros del experimento
# 	experiment <- 1

# 	pathTrainImages <- "../data/yemas-cut/trainset-base/"
# 	pathTrainsetCSV <- paste(pathTrainImages, "trainset-base.csv", sep="")

# 	pathOutputDict <- "../data/learning_curves/dictionaries/"
# 	pathOutputHist <- "../data/learning_curves/LC-histograms-test/"

# 	dictSize <- seq(150,650,50)
# 	cvFoldTrainset <- 5
# 	classBud <- 1

# 	Dmin <- 10
# 	Dstep <- 40


# 	# Lee el dataset de entrenamiento
# 	trainset <- read.csv(pathTrainsetCSV, header=T)

# 	# Info del dataset para hacer el split
# 	D <- nrow(trainset)
# 	trainsetsSize <- seq(Dmin, D, Dstep)

# 	# Procesa el trainset para correr los experimentos
# 	imgFiles <- paste(pathTrainImages, trainset$ImagenName, sep="")
# 	imgClass <- trainset$class
# 	idxClassBud <- which(imgClass==classBud)
# 	idxClassNoBud <- which(imgClass!=classBud)
# 	ratioSplit <- length(idxClassBud)/D


# 	# Bucle principal para los diferentes tamaños de datasets
# 	for (ts in trainsetsSize) {

# 		# Samplea 'cvFoldTrainset' subsets de 'ts' datapoints cada uno
# 		for (cv in 1:cvFoldTrainset) {
# 			tempIdxBud <- sample( idxClassBud, round( ts*ratioSplit ) )
# 			tempIdxNoBud <- sample( idxClassNoBud, round( ts*(1-ratioSplit) ) )

# 			# Siempre se mantiene la misma relación entre cantidad de datapoints yema y no-yema
# 			cvIdxTrainset <- c(tempIdxBud, tempIdxNoBud)

# 			cvImgFiles <- paste(imgFiles[cvIdxTrainset], collapse=",")

# 			temp <- c( rep(TRUE, round( ts*ratioSplit )), rep(FALSE, round( ts*(1-ratioSplit) )) )
# 			mClass <- matrix(temp, ncol=1)
# 			colnames(mClass) <- c("class")

# 			# Itera sobre los tamaños de diccionario
# 			for (ds in dictSize) {

# 				outputDict <- paste(pathOutputDict, "dict-s", ds, "-D", ts, "-cv", cv, sep="")
# 				outputHist <- paste(pathOutputHist, "hist-s", ds, "-D", ts, "-cv", cv, sep="")
# 				cdmString <- paste("../Debug/BoW --exp ", experiment, " --imgList '", cvImgFiles,"'",
# 					" --dictSize ", ds," --outDict '", outputDict,"' --outHist '", outputHist,"'", sep="")
# 				try(system(cdmString))

# 				outputHist <- paste(pathOutputHist, "hist-s", ds, "-D", ts, "-cv", cv, ".csv", sep="")

# 				imgHist <- read.csv(outputHist, header=FALSE, sep=",", stringsAsFactors=FALSE)
# 				imgHist <- cbind(imgHist, mClass)

# 				write.csv(imgHist, outputHist)
# 			}

# 		}

# 	}

# }


# expLC2 <- function() {
# 	# Parametros del experimento
# 	experiment <- 2

# 	pathTrainImages <- "../data/yemas-cut/trainset-base/"
# 	pathTrainsetCSV <- paste(pathTrainImages, "trainset-base.csv", sep="")

# 	pathTestImages <- "../data/yemas-cut/testset-base/"
# 	pathTestsetCSV <- paste(pathTestImages, "testset-base.csv", sep="")

# 	pathOutputDict <- "../data/learning_curves/dictionaries/"
# 	pathOutputHist <- "../data/learning_curves/LC-histograms-test/"

# 	dictSize <- seq(150,650,50)
# 	cvFoldTestset <- 5
# 	classBud <- 1

# 	Dmin <- 10
# 	Dstep <- 40


# 	# Lee el dataset de entrenamiento
# 	trainset <- read.csv(pathTrainsetCSV, header=T)

# 	# Lee el dataset de testeo
# 	testset <- read.csv(pathTestsetCSV, header=T)

# 	# Info del dataset para hacer el split
# 	D <- nrow(trainset)
# 	trainsetsSize <- seq(Dmin, D, Dstep)

# 	# Procesa el trainset para correr los experimentos
# 	imgFiles <- paste(pathTestImages, testset$ImagenName, sep="")
# 	imgFiles <- paste(imgFiles, collapse=",")
# 	imgClass <- testset$class

# 	# Bucle principal para los diferentes tamaños de datasets
# 	for (ts in trainsetsSize) {

# 		# Samplea 'cvFoldTestset' subsets de 'ts' datapoints cada uno
# 		for (cv in 1:cvFoldTestset) {

# 			# Itera sobre los tamaños de diccionario
# 			for (ds in dictSize) {

# 				outputDict <- paste(pathOutputDict, "dict-s", ds, "-D", ts, "-cv", cv, sep="")
# 				outputHist <- paste(pathOutputHist, "hist-s", ds, "-D", ts, "-cv", cv, sep="")
# 				cdmString <- paste("../Debug/BoW --exp ", experiment, " --imgList '", imgFiles,"'",
# 					" --dictSize ", ds," --outDict '", outputDict,"' --outHist '", outputHist,"'", sep="")
# 				try(system(cdmString))

# 				outputHist <- paste(pathOutputHist, "hist-s", ds, "-D", ts, "-cv", cv, ".csv", sep="")

# 				imgHist <- read.csv(outputHist, header=FALSE, sep=",", stringsAsFactors=FALSE)
# 				imgHist <- cbind(imgHist, as.logical(imgClass) )

# 				write.csv(imgHist, outputHist)

# 			}

# 		}

# 	}
# }


# expLC3 <- function() {
# 	# Parametros del experimento
# 	pathOutputLC <- "../../bdml_svm_ocsvm_BoW/output/LC-binary/outputLC.csv"
# 	outputLC <- read.csv(pathOutputLC, header=TRUE, sep=",", stringsAsFactors=FALSE)
# 	pathBestParms <- "../../learning-curves/trainset/best_parameters_binary.csv"
# 	bestParams <- read.csv(pathBestParms, header=TRUE, sep=",", stringsAsFactors=FALSE)

# 	dictSize <- seq(150,650,50)
# 	cvFoldTrainset <- 5
# 	algSVM <- c("c-svm", "oc-svm")

# 	Dmin <- 10
# 	Dstep <- 40
# 	D <- 690
# 	trainsetsSize <- seq(Dmin, D, Dstep)


# 	idxTrain <- "train"==outputLC$type
# 	idxTest <- "test"==outputLC$type

# 	# Itera sobre los algoritmos SVM
# 	for (alg in algSVM) {

# 		idxAlg <- alg==outputLC$alg

# 		# Itera sobre los tamaños de diccionario
# 		for (ds in dictSize) {

# 			idxDS <- ds==outputLC$s

# 			idxBestParam <- bestParams$s==ds & bestParams$algorithm==alg

# 			cost <- bestParams[idxBestParam,"cost"]
# 			gamma <- bestParams[idxBestParam, "gamma"]
# 			nu <- bestParams[idxBestParam, "nu"]

# 			trainLine <- NULL
# 			trainPrecLine <- NULL
# 			trainRecLine <- NULL
# 			testLine <- NULL
# 			testPrecLine <- NULL
# 			testRecLine <- NULL

# 			# Bucle principal para los diferentes tamaños de datasets
# 			for (ts in trainsetsSize) {

# 				idxTS <- ts==outputLC$D

# 				idxTempTrain <- idxAlg & idxDS & idxTS & idxTrain
# 				idxTempTest <- idxAlg & idxDS & idxTS & idxTest

# 				trainError <- 1 - mean(outputLC[idxTempTrain,]$Accuracy)
# 				trainLine <- rbind(trainLine, c(ts,trainError) )

# 				testError <- 1 - mean(outputLC[idxTempTest,]$Accuracy)
# 				testLine <- rbind(testLine, c(ts,testError) )

# 				trainPrecError <- 1 - mean(outputLC[idxTempTrain,]$Accuracy)
# 				trainPrecLine <- rbind(trainPrecLine, c(ts,trainPrecError) )

# 				testRecError <- 1 - mean(outputLC[idxTempTest,]$Accuracy)
# 				testRecLine <- rbind(testRecLine, c(ts,testRecError) )

# 			}

# 			graphName <- paste("../../bdml_svm_ocsvm_BoW/output/LC-binary/LC-graphics/graph-s", ds, "-alg_", alg, ".jpg", sep="")

# 			jpeg( graphName, width = 800, height = 800 )

# 			title <- ""
# 			if (alg=="c-svm")
# 				title <- paste("Learning Curves: dictSize ", ds, "; alg ", alg, " (cost: ", cost, ", gamma: ", gamma,")", sep="")
# 			else
# 				title <- paste("Learning Curves: dictSize ", ds, "; alg", alg, " (cost: ", cost, ", nu: ", nu,")", sep="")

# 			plot(rbind(testLine, trainLine), pch=19, cex=1.5, col="darkgray", main=title, xlab="D", ylab="Error")
# 			lines(trainLine, col="blue", lwd=3)
# 			lines(testLine, col="red", lwd=3)

# 			legend("topright", legend = c("Train Error", "Test Error"),
#                     lwd=3, inset=.01, col=c("blue", "red"))

# 			dev.off()

# 		}
# 	}

# }


