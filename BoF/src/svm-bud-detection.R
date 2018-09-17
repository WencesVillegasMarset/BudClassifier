#options(error = recover)
require(compiler)
enableJIT(3)

library("e1071") # implementacion de libSVM
#library("caret") # Classification and Regression Training
#library("mlearning") # Machine learning algorithms with unified interface and confusion matrices
# library(doMC) # Multicore
# cores <- 8 # Number of cores
# registerDoMC(cores) # register cores
#setwd("~/IA/bdml/src")
#source("./commons.R")


# dir_input_train <- "../../learning-curves/trainset/"
# dir_input_test <- "../../learning-curves/testset/"
# dir_output <- "../output/LC-binary/"

dir_input_train <<- NULL
dir_input_test <<- NULL
dir_output <<- NULL
dir_RunTime <<- NULL

set.directories <- function(input_train, input_test, output, output_rt) {

  dir_input_train <<- input_train
  dir_input_test <<- input_test
  dir_output <<- output
  dir_RunTime <<- output_rt

  if (!file.exists(dir_output)) dir.create(dir_output)

}

##TODO: borrar esta funcion e incluirla en la funcion general
##Esta funcion es para levantar los tune.txt y obtener el mejor sobre todos los diccionarios
get_tune_results <- function(){
  tunesDir <- outputSVM
  dictSize <- dictSize
  iterations <- iterations
  ratios <- general.ratios
  tunesFiles <- list.files(tunesDir,"tune.txt")
  tune.result <- matrix(0,ncol=6)
  colnames(tune.result) <- c("dic","ratio","it","gamma","cost","fmeasure")
  tune.result <- tune.result[-1,]

  for (ds in dictSize){
    for (r in ratios){
      gammas <- NULL
      costs <- NULL
      fmeasures <- NULL
      for (i in iterations){
        tFName <- paste(tunesDir,"s",ds,"_r",r,"_it",i,"_csvm_tune.txt",sep="")
        if (file.exists(tFName)){
	  tunesData <- read.csv(tFName, sep="\t",header=FALSE)
	  tD <- unlist(strsplit(as.character(tunesData[4,1])," "))
     	  values <- as.numeric(tD[which(tD > 0)])
      	  tune.result <- rbind(tune.result,c(ds,r,i,values[1],values[2],values[3]))
        }else{
	  cat(paste("\nNo existe el archivo:",tFName))
        }
      }
    }
  }
  res.tune <- paste(tunesDir,"res.tune.csv",sep="")
  write.table(tune.result,file=res.tune,sep=",",row.names=FALSE)
  tune.result
}

##Esta funcion la utilizo para maximizar la fmeasure
my.error.function <- function(truth,pred){
  cat(".")
  pred <- factor(pred,levels=c("YEMA","NOYEMA"))
  truth <- factor(truth,levels=c("YEMA","NOYEMA"))
  datos <- prf(pred, truth)
  res <- datos$stats[1,4]
  return(res)
}

get.proportional.kfolds <- function(mydataset,class.name,k,cw){

  folds <- list()
  for(i in 1:k) folds[[i]] <- data.frame()

  # obtengo las diferentes clases
  classes <- unique(mydataset[,class.name])

  #Calculo la longitud del dataset
  N <- nrow(mydataset)

  #Longitud de cada fold
  rf <- floor(N/k)

  #Longitud de cada clase
  rc <- floor(rf/length(classes))

  #Desordeno el dataset
  mydataset <- mydataset[sample(N),]

  for (c in classes){
    if (cw==1){
      #Tomo los ejemplos de la clase c
      subsetC <- mydataset[which(mydataset[,class.name]==c),]
    }else{
      #Tomo solo los ejemplos que no se repiten y de la clase c
      subsetC <- unique(mydataset[which(mydataset[,class.name]==c),])
    }
    n <- nrow(subsetC)
    for(i in 1:k) {
      first = 1 + (((i - 1) * n) %/% k)
      last = ((i * n) %/% k)
      if (cw==1){
        #tomo los ejemplos de la clase c correspondientes a la particion k
        folds[[i]] <- rbind(folds[[i]],subsetC[first:last,])
      }else{
        #tomo la cantidad de muestras rc de la clase c para balancear y las agrego a la particion k
        balancedSamples <- sample(first:last,rc,replace=TRUE)
        folds[[i]] <- rbind(folds[[i]],subsetC[balancedSamples,])
      }
    }
  }

  return(folds)
}

##
get.best.parameters.c_svm <- function(mydataset,prefijo,save=TRUE,balancedTest=TRUE,c.weights=NULL,cw=-1){
  timestamp()
  gammas <- 2^seq(-14,-7,1) #seq(-14,14,1)
  costs <- 2^seq(5,14,1) #seq(-14,14,1)
  k <- 5
  type <- "C-classification"
  filename <- paste(dir_output,prefijo,"_csvm_tune.RData",sep="")

  if (save || !file.exists(filename)){
    #     c_svm_tune <- tune.svm(class~., data=mydataset,
    #                            gamma = gammas,
    #                            cost = costs,
    #                            type = type,
    #                            tunecontrol = tune.control(sampling = "cross", cross=k, error.fun = my.error.function )
    #     )
    data.folds <- get.proportional.kfolds(mydataset,"class",k,cw)

    fmeasure.mean.max <- NULL

    foreach(g=1:length(gammas) ) %dopar% {  # multicore
    # for (g in 1:length(gammas)){ #single core
      gamma <- gammas[g]
      for (cost in costs){

        # Si el archivo existe omite el computo (en caso de ser necesario recuperarse de alguna caida del script)
        file_pattern <- paste("^", prefijo,
                           "_gamma_",gamma,
                           "_cost_",cost,sep="")
        if ( length( list.files(dir_output, file_pattern) ) > 0 ) next

        fmeasure.mean <- 0
        for(i in 1:k){
          trainset <- NULL
          testset <- NULL
          for(j in 1:k){
            if(j!=i){
              trainset <- rbind(trainset,data.folds[[j]])
            }else{
              if (balancedTest){
                indx.yemas <- which(data.folds[[j]][,"class"]=="YEMA")
                tmp.yemas <- data.folds[[j]][indx.yemas,]
                tmp.noyemas <- data.folds[[j]][-indx.yemas,]
                n.yemas <- nrow(tmp.yemas)
                testset <- rbind(tmp.yemas,tmp.noyemas[sample(n.yemas),])
              }else{
                testset <- data.folds[[j]]
              }
            }
          }

          k.model <- svm(class~., data=trainset,
                         gamma = gamma,
                         cost = cost,
                         type = type,
                         class.weights=c.weights)
          res <- results.c_svm <- predict(k.model,
                                          newdata=testset[,-ncol(testset)]
          )
          fmeasure.mean <- fmeasure.mean + my.error.function(testset[,ncol(testset)],res)
        }

        fmeasure.mean <- fmeasure.mean / k

        # Aqui guardo parameters <- gamma,cost,fmeasure.mean en un archivo de disco
        cmdstring <- paste("touch ",
                           dir_output,
                           prefijo,
                           "_gamma_",gamma,
                           "_cost_",cost,
                           "_fm_",fmeasure.mean,
                           "_.tune",sep="")
        system(cmdstring)

        # cat(paste("\nDic:",prefijo,"Gamma:",gamma,"Cost:",cost,"F1:",fmeasure.mean,sep=" "))
        cat("/")
      }
    }
    # Busco los mejores parametros
    parameters <- NULL
    c_svm_tune <- list()

    # leer de disco los archivos .tune
    fl <- list.files(dir_output,"*.tune$")
    for (f in fl){
      p <- unlist(strsplit(f,"_"))
      gamma <- p[5]
      cost <- p[7]
      fmeasure.mean <- p[9]
      parameters <- rbind(parameters,matrix(c(gamma,cost,fmeasure.mean),ncol=3))

      if ( fmeasure.mean > fmeasure.mean.max || is.null(fmeasure.mean.max)){
        fmeasure.mean.max <- fmeasure.mean
        c_svm_tune$best.parameters[["gamma"]] <- gamma
        c_svm_tune$best.parameters[["cost"]] <- cost
        c_svm_tune$best.parameters[["FMeasure"]] <- fmeasure.mean
      }
    }

    colnames(parameters) <- c("gamma","cost","Fmeasure")

    # borro los archivos .tune
    cmdString <- paste("rm ",dir_output,"*.tune",sep="")
    system(cmdString)

    # guardo los mejores parametros
    save(c_svm_tune, parameters, file=filename, compress="bzip2")
  }else{
    load(filename)
  }

  #   cat("\nParameters Search:\n")
  #   print(parameters)
  sink(file=paste(dir_output,prefijo,"_csvm_tune.txt",sep=""))
  cat("\nParameters Search:\n")
  cat("\nBest Parameters:\n")
  print(c_svm_tune$best.parameters)
  print(parameters)
  sink(NULL)
  return(c_svm_tune)
}

get.best.parameters.oc_svm <- function(mydataset, prefijo, save=TRUE){
  #   gammas <- 2^(-1:1)
  #   costs <- 2^(-1:1)
  timestamp()
  gammas <- 2^(-15:3)
  nus <- seq(from=0.05,to=0.95,by=0.05)
  k <- 5
  type <- "one-classification"
  filename <- paste(dir_output,prefijo,"_ocsvm_tune.RData",sep="")

  if (save || !file.exists(filename)){
    oc_svm_tune <- tune.svm(class~., data=mydataset,
                            gamma = gammas,
                            nu = nus,
                            type=type,
                            tunecontrol = tune.control(sampling = "cross", cross=k, error.fun = my.error.function)
    )
    save(oc_svm_tune,file=filename,compress="bzip2")
  }else{
    load(filename)
  }

  print(summary(oc_svm_tune))
  sink(file=paste(dir_output,prefijo,"_ocsvm_tune.txt",sep=""))
  print(summary(oc_svm_tune))
  sink(NULL)
  return(oc_svm_tune)
}

c_svm <- function(mydataset, bp.c_svm, prefijo, save=TRUE, c.weights=NULL){
  gamma <- bp.c_svm$best.parameters[["gamma"]]
  cost <- bp.c_svm$best.parameters[["cost"]]
  type <- "C-classification"
  filename <- paste(dir_output,prefijo,"_csvm_bestmodel.RData",sep="")

  if (save || !file.exists(filename)){
    c_svm_bestmodel <- svm(class~., data=mydataset,
                           gamma = gamma,
                           cost = cost,
                           type=type,
                           probability = TRUE,
                           class.weights=c.weights)
    save(c_svm_bestmodel,file=filename,compress="bzip2")
  }else{
    load(filename)
  }

  print(summary(c_svm_bestmodel))
  return(c_svm_bestmodel)
}

oc_svm <- function(mydataset, bp.oc_svm, prefijo, save=TRUE){
  gamma <- bp.oc_svm$best.parameters[["gamma"]]
  nu <- bp.oc_svm$best.parameters[["nu"]]
  type <- "one-classification"
  filename <- paste(dir_output,prefijo,"_ocsvm_bestmodel.RData",sep="")

  if (save || !file.exists(filename)){
    oc_svm_bestmodel <- svm(class~., data=mydataset,
                            gamma = gamma,
                            nu=nu,
                            type=type)
    save(oc_svm_bestmodel,file=filename,compress="bzip2")
  }else{
    load(filename)
  }
  # print(summary(oc_svm_bestmodel))
  return(oc_svm_bestmodel)
}

## Precision - P = TP/(TP+FP) how many idd actually success/failure
## Recall - R = TP/(TP+FN) how many of the successes correctly idd
## F-score - F = (2 * P * R)/(P + R) harm mean of precision and recall
prf <- function(preds,trues){
  k <- 4 # decimals
  clss <- c("YEMA","NOYEMA")
  preds <- factor(preds,levels=clss)
  trues <- factor(trues,levels=clss)
  if (any(is.na(preds))){
    cm=matrix(NA,ncol=length(clss),nrow=length(clss),dimnames=list(clss,clss))
  }else{
    cm <- table(preds, trues, dnn=c("Predict","Truth"))
  }
  rmat.names <- c("Accuracy")
  rmat <- matrix(NA,ncol =(1+(length(clss)*3)),nrow=1)
  rmat[1,1] <- round(sum(cm[1,1],cm[2,2])/sum(cm),k) # Accuracy
  for (cl in clss){
    rmat.names <- c(rmat.names,paste("Prec",cl,sep='-'),paste("Rec",cl,sep='-'),paste("F-Scr",cl,sep='-'))
    rmat.ncol <- length(rmat.names)
    prec <- round(cm[cl,cl]/sum(cm[cl,]) ,k)#Precision
    rmat[1,rmat.ncol-2] <- ifelse(is.nan(prec),0,prec)
    reca <- round(cm[cl,cl]/sum(cm[,cl]) ,k)#Recall
    rmat[1,rmat.ncol-1] <- ifelse(is.nan(reca),0,reca)
    fmeasu <- round((2*rmat[1,rmat.ncol-2]*rmat[1,rmat.ncol-1])/(rmat[1,rmat.ncol-2]+rmat[1,rmat.ncol-1]),k) #F1Measure
    rmat[1,rmat.ncol] <-  ifelse(is.nan(fmeasu),0,fmeasu)
  }
  colnames(rmat) <- rmat.names
  return(list(cm=cm,stats=rmat))
}

generate.results <- function(results, ratio=-1, it=0, cw=-1){

  pref <- ""
  if (ratio!=-1) pref <- paste0(pref, "r", ratio, "_")
  if (it!=0) pref <- paste0(pref, "it", it, "_")
  if (cw!=-1) pref <- paste0(pref, "cw", cw, "_")

  file_res <- paste(dir_input_test,"res_",pref,format(Sys.time(), "%Y-%m-%d_%H.%M.%S"),".txt",sep="")
  sink(file=file_res)
  indice <- 1
  for ( pref in results$prefijos){
    for (i in 1:length(results$algorithm.names)){
      cat(paste("\nSummary: ",results$algorithm.names[i], "Dataset: " , pref , "\n"))
      output <-(prf(results$results[[indice]],results$true.values))
      cat("\nConfusion Matrix:\n")
      print(output$cm)
      cat("\nStats:\n")
      print(output$stats)
      indice <- indice + 1
    }
  }

  cat("\nResume:\n")
  for (cl in 1:length(levels(results$true.values))){
    cat(paste("\n Class: ", (levels(results$true.values))[cl],"\n"))
    res.mat <- matrix(NA,
                      ncol=6,
                      nrow=(length(results$prefijos)*length(results$algorithm.names)),
                      dimnames = list(c(),c("Dataset","Algorithm","Accuracy","Precision","Recall","Fscore")))
    indice <- 1
    for (p in 1:length(results$prefijos)){
      for (i in 1:length(results$algorithm.names)){
        output <-(prf(results$results[[indice]],results$true.values))
        res.mat[indice,1] <- results$prefijos[p]
        res.mat[indice,2] <- results$algorithm.names[i] #Algorithm
        res.mat[indice,3] <- output$stats[1,1] #Accuracy
        res.mat[indice,4] <- output$stats[1,((cl-1)*3)+2] #Precision
        res.mat[indice,5] <- output$stats[1,((cl-1)*3)+3] #Recall
        res.mat[indice,6] <- output$stats[1,((cl-1)*3)+4] #Fscore
        indice <- indice + 1
      }
    }
    print(res.mat)
  }
  sink(NULL)
  results.data <- readLines(file_res)
  for ( line in results.data) cat(paste(line,"\n"))
}


appendToFile <- function(line,filename){
  sink(file=filename,append=TRUE)
  cat(paste(line,"\n"))
  sink(NULL)
}

## Funcion para entrenar el modelo con los mejores parametros
# Devuelve el mejor modelo
# mydataset = dataset de entrenamiento
# algorithm = indice del algoritmo a utilizar: C-SVM=1 OC-SVM=2
# prefijo = prefijo utilizado para almacenar los datos calculados i.e.: ds1 > ds1-csvm_bestmodel.txt , ds1-ocsvm_tune.txt
train.model <- function(mydataset=NULL, algorithm=1, prefijo=NULL, save=TRUE, c.weights=NULL, cw=-1){
  # set.seed(1)
  result.model <- NULL
  mydataset <- mydataset[sample(nrow(mydataset)),] # desordeno los datos


  if ( algorithm == 1) {
    ## C-SVM
    cat("\nStart getting best model C-SVM\n")
    bp.c_svm <- get.best.parameters.c_svm(mydataset, prefijo,
                                            balancedTest=FALSE,save=save,
                                            c.weights=c.weights,cw=cw)
    # bp.c_svm <- list()
    # bp.c_svm$best.parameters[["gamma"]] <- 0.0078125
    # bp.c_svm$best.parameters[["cost"]] <- 2048
    result.model <- c_svm(mydataset, bp.c_svm, prefijo, save=save, c.weights=c.weights)
  }

  # if ( algorithm == 2){
  #   ## OC-SVM
  #   cat("\nStart getting best model OC-SVM\n")
  #   bp.oc_svm <- get.best.parameters.oc_svm(mydataset, prefijo ,save=save)
  #   result.model <- oc_svm(mydataset, bp.oc_svm, prefijo, save=save)
  # }
  return(result.model)
}


#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################


## Funcion para correr el experimento
run.bdml.experiment <- function(dic_pattern="", ratio=-1, it=0, cw=-1) {

  if(is.null(dir_input_train) || is.null(dir_input_test) || is.null(dir_output) ) {
    stop("Falta settear los directorios")
  }

  ds <- dic_pattern
  dic_pattern <- paste0("s",ds)

  runtimeTrain <- system.time({

    dic_pattern <- paste0(dic_pattern, ".csv")

    # Leo los histogramas para train
    train.folder <- dir_input_train
    train.data.files <- list.files(train.folder, dic_pattern)
    tmp.pref <- unlist(strsplit(unlist(strsplit(train.data.files,".csv")),"-"))
    datasets.pref <- tmp.pref[which(tmp.pref != "hist")]

    prefRIT <- ""
    if (ratio!=-1) prefRIT <- paste0(prefRIT, "_r", ratio)
    if (it!=0) prefRIT <- paste0(prefRIT, "_it", it)
    # if (cw!=-1) prefRIT <- paste0(prefRIT, "_cw", cw) la comento porque trae problemas
    datasets.pref <- paste0(datasets.pref, prefRIT)

    train.data.files <- paste0(train.folder,train.data.files)

    # Leo los histogramas para test
    test.folder <- dir_input_test
    test.data.files <- list.files(test.folder, dic_pattern)
    test.data.files <- paste0(test.folder,test.data.files)

    datasets <- list()
    for(i in 1:length(train.data.files)){
      datasets[[i]] <- list()
    }

    for (i in 1:length(train.data.files)){

      # leo el dataset de datos train
      mydataset <- read.csv(train.data.files[i],
                            header=FALSE,
                            sep=",",
                            stringsAsFactors=FALSE)

      colnames(mydataset)[ncol(mydataset)] <- "class"
      rownames(mydataset) <- 1:(nrow(mydataset))

      if (ratio!=-1) {
        tempPosData <- mydataset[mydataset$class,]

        tempNegData <- mydataset[!mydataset$class,]

        idxData <- sample(1:nrow(tempNegData), nrow(tempPosData)*ratio )
        tempNegData <- tempNegData[idxData,]


        if (cw == 1){
          c.weights <- c("YEMA"=nrow(tempNegData), "NOYEMA"=nrow(tempPosData))/(nrow(tempPosData)+nrow(tempNegData))
        }else{
          replace <- FALSE; if (ratio>1) replace <- TRUE
          idxData <- sample(1:nrow(tempPosData), nrow(tempPosData)*ratio, replace=replace)
          tempPosData <- tempPosData[idxData,]
          c.weights <- NULL
        }

        mydataset <- rbind(tempPosData, tempNegData)

        # cat("\nnrow(trueData) = ", nrow(tempPosData),"\tnrow(falseData) = ", nrow(tempNegData))
        # cat("\nc.weights = ")
        # print(c.weights)

      }

      #     ## 70% of the sample size
      #     smp_size <- floor(0.7 * nrow(mydataset))#
      #     ## set the seed to make your partition reproductible
      #     set.seed(123)
      #     train_ind <- sample(seq_len(nrow(mydataset)), size = smp_size)

      #traindataset <- mydataset[train_ind,-c(1,2)] # elimino columnas sobrantes ID, nombre
      traindataset <- mydataset[,-1] # utilizo todo el dataset

      # leo el dataset de datos
      mydataset <- read.csv(test.data.files[i],
                            header=FALSE,
                            sep=",",
                            stringsAsFactors=FALSE)

      colnames(mydataset)[ncol(mydataset)] <- "class"
      rownames(mydataset) <- 1:(nrow(mydataset))

      #testdataset <- mydataset[-train_ind,-c(1,2)] # elimino columnas sobrantes ID, nombre
      testdataset <- mydataset[,-1] # utilizo todo el dataset
      testfilename <- mydataset[,1]

      #     ####tratamiento NA en train
      #     train.na.index <- NULL
      #     for (j in 1:col(traindataset)){
      #       train.na.index <- c(train.na.index,which(traindataset[,j]==" NA"))
      #       train.na.index <- c(train.na.index,which(is.na(traindataset[,j])))
      #     }
      #     train.na.index <- sort(unique(train.na.index))
      #     # los reemplazo con 0
      #     # traindataset[train.na.index,]<- 0
      #     # los elimino
      #     if (length(train.na.index)!=0) {
      #       traindataset <- traindataset[-train.na.index,]
      #     }
      #
      #
      #     #tratamiento NA en test
      #     test.na.index <- NULL
      #     for (j in 1:ncol(testdataset)){
      #       test.na.index <- c(test.na.index,which(testdataset[,j]==" NA"))
      #       test.na.index <- c(test.na.index,which(is.na(testdataset[,j])))
      #     }
      #     test.na.index <- sort(unique(test.na.index))
      #     # los reemplazo con 0
      #     # testdataset[test.na.index,]<- 0
      #     # los elimino
      #     if (length(train.na.index)!=0){
      #       testdataset <- testdataset[-test.na.index,]
      #       testfilename <- testfilename[-test.na.index]
      #     }

      traindataset <- as.data.frame(data.matrix(traindataset))
      if ( any( is.na(traindataset[,"class"]) ) )
        traindataset <- traindataset[-which(is.na(traindataset[,"class"])),]
      traindataset[,ncol(traindataset)] <- ifelse(traindataset[,ncol(traindataset)]==1,"YEMA","NOYEMA")

      testdataset <- as.data.frame(data.matrix(testdataset))
      if ( any( is.na(testdataset[,"class"]) ) )
        testdataset <- testdataset[-which(is.na(testdataset[,"class"])),]
      testdataset[,ncol(testdataset)] <- ifelse(testdataset[,ncol(testdataset)]==1,"YEMA","NOYEMA")

      datasets[[i]]$ds <- traindataset
      datasets[[i]]$ts <- testdataset
      datasets[[i]]$prefijo <- datasets.pref[i]
      datasets[[i]]$testfilename <- testfilename
    }

    ## TODO: delete se movio el multicore a los gamma
    # foreach(i=1:length(train.data.files) ) %dopar% {  # multicore

    for ( i in 1:length(train.data.files)){ # single core
      # train.model C-SVM
      cat(paste("Train C-SVM" , train.data.files[i]))

      res <- try(train.model(mydataset=datasets[[i]]$ds,
                             algorithm=1,
                             prefijo=datasets[[i]]$prefijo,
                             save=FALSE,
                             c.weights=c.weights,
                             cw=cw))
      if(inherits(res,"try-error")){
        cat("-----------------error\n")
      }

      #  # train.model OC-SVM
      # cat(paste("Train OC-SVM" , train.data.files[i]))

      # #entreno solo con las yemas ??????
      # train_SY <- datasets[[i]]$ds[which(datasets[[i]]$ds[,'class']==TRUE),]

      # res <- try(train.model(mydataset=train_SY,
      #             algorithm=2,
      #             prefijo=datasets[[i]]$prefijo,
      #             save=FALSE))
      # if(inherits(res,"try-error")){
      #             cat("-----------------error\n")
      #             }

    }
  })

  experiment <- "runtimeTrainSVM"
  rtvar <- runtimeTrain
  outRT <- paste0(  dir_RunTime, "s_",ds,"_",experiment,
                    "_user_",rtvar[1],
                    "_system_",rtvar[2],
                    "_elapsed_",rtvar[3],
                    ".runtime")
  write(rtvar,outRT)


  testData <- NULL
  predictions <- NULL

  runtimeTest <- system.time({
    predictions <- list()
    #foreach( i=1:length(test.data.files) ) %dopar% {  # multicore
    for ( i in 1:length(test.data.files)) {           # singlecore
      prefijo <- datasets[[i]]$prefijo
      testData <- datasets[[i]]$ts

      results.c_svm <- NULL
      filename <- paste(dir_output,prefijo,"_csvm_bestmodel.RData",sep="")
      if (file.exists(filename)){
        load(filename)
        results.c_svm <- predict(c_svm_bestmodel,
                                 newdata=testData[,-ncol(testData)],
                                 probability = TRUE)
        results.c_svm.prob <- attr(results.c_svm, "probabilities")
        results.c_svm <- as.matrix(results.c_svm)
        results.c_svm <- factor(results.c_svm,levels=c("YEMA","NOYEMA"))

        pred.file <- paste(dir_input_test,prefijo,"_c_svm_predict.csv",sep="")
        x <- cbind(imageName=datasets[[i]]$testfilename, pred=as.character(results.c_svm),
                   probTrue=results.c_svm.prob[,1])
        cat(paste("Saving:",pred.file,"\n"))

        write.table(x,file=pred.file,sep=",", row.names=FALSE)
      }else{
        results.c_svm <- NA
      }

      results.oc_svm <- NULL
      filename <- paste(dir_output,prefijo,"_ocsvm_bestmodel.RData",sep="")
      if (file.exists(filename)){
        load(filename)
        results.oc_svm <- predict(oc_svm_bestmodel,
                                  newdata=testData[,-ncol(testData)])
        results.oc_svm <- factor(results.oc_svm,levels=c("YEMA","NOYEMA"))
        pred.file <- paste(dir_output,prefijo,"_oc_svm_predict.csv",sep="")
        x <- cbind(imageName=datasets[[i]]$testfilename, pred=as.character(results.oc_svm))
        cat(paste("Saving:",pred.file,"\n"))
        write.table(x,file=pred.file,sep=",", row.names=FALSE)
      }else{
        results.oc_svm <- NA
      }

      predictions[[(i*2)-1]] <- results.c_svm
      predictions[[i*2]] <- results.oc_svm
    }
  })

  experiment <- "runtimeTestSVM"
  rtvar <- runtimeTest
  outRT <- paste0(  dir_RunTime, "s_",ds,"_",experiment,
                    "_user_",rtvar[1],
                    "_system_",rtvar[2],
                    "_elapsed_",rtvar[3],
                    ".runtime")
  write(rtvar,outRT)

  truthvalues <- factor(testData[,ncol(testData)],levels=c("YEMA","NOYEMA"))

  ## RESULTS
  cat("\nRESULTS\n")
  results <- list(results=predictions,
                  algorithm.names=c("C-SVM", "OC-SVM"),
                  prefijos=datasets.pref,
                  true.values=truthvalues)

  generate.results(results, ratio, it, cw)


  # cat("\nTRAINING RUNTIME (SVM):\n")
  # print(runtimeTrain)

  # cat("\nTESTING RUNTIME (SVM):\n")
  # print(runtimeTest)

}

## Funcion para correr el experimento
run.windowed <- function(){

  if(is.null(dir_input_train) || is.null(dir_input_test) || is.null(dir_output) ) {
    stop("Falta settear los directorios")
  }

  # Leo los histogramas para train
  train.folder <- dir_input_train
  train.data.files <- list.files(train.folder)
  tmp.pref <- unlist(strsplit(unlist(strsplit(train.data.files,".csv")),"-"))
  datasets.pref <- tmp.pref[which(tmp.pref != "hist")]
  train.data.files <- paste0(train.folder,train.data.files)

  # Leo los histogramas para test
  test.folder <- dir_input_test
  test.data.files <- list.files(test.folder)
  test.data.files <- paste0(test.folder,test.data.files)

  datasets <- list()
  for(i in 1:length(train.data.files)){
    datasets[[i]] <- list()
  }

  for (i in 1:length(train.data.files)){

    # leo el dataset de datos train
    mydataset <- read.csv(train.data.files[i],
                          header=FALSE,
                          sep=",",
                          stringsAsFactors=FALSE)

    colnames(mydataset)[ncol(mydataset)] <- "class"
    rownames(mydataset) <- 1:(nrow(mydataset))

    #     ## 70% of the sample size
    #     smp_size <- floor(0.7 * nrow(mydataset))#
    #     ## set the seed to make your partition reproductible
    #     set.seed(123)
    #     train_ind <- sample(seq_len(nrow(mydataset)), size = smp_size)

    #traindataset <- mydataset[train_ind,-c(1,2)] # elimino columnas sobrantes ID, nombre
    traindataset <- mydataset[,-1] # utilizo todo el dataset



    # leo el dataset de datos
    mydataset <- read.csv(test.data.files[i],
                          header=FALSE,
                          sep=",",
                          stringsAsFactors=FALSE)

    #agrego class
    mydataset <- cbind(mydataset,0)
    colnames(mydataset)[ncol(mydataset)] <- "class"
    rownames(mydataset) <- 1:(nrow(mydataset))

    #testdataset <- mydataset[-train_ind,-c(1,2)] # elimino columnas sobrantes ID, nombre
    testdataset <- mydataset[,-1] # utilizo todo el dataset
    testfilename <- mydataset[,1]

    #     ####tratamiento NA en train
    #     train.na.index <- NULL
    #     for (j in 1:col(traindataset)){
    #       train.na.index <- c(train.na.index,which(traindataset[,j]==" NA"))
    #       train.na.index <- c(train.na.index,which(is.na(traindataset[,j])))
    #     }
    #     train.na.index <- sort(unique(train.na.index))
    #     # los reemplazo con 0
    #     # traindataset[train.na.index,]<- 0
    #     # los elimino
    #     if (length(train.na.index)!=0) {
    #       traindataset <- traindataset[-train.na.index,]
    #     }
    #
    #
    #     #tratamiento NA en test
    #     test.na.index <- NULL
    #     for (j in 1:ncol(testdataset)){
    #       test.na.index <- c(test.na.index,which(testdataset[,j]==" NA"))
    #       test.na.index <- c(test.na.index,which(is.na(testdataset[,j])))
    #     }
    #     test.na.index <- sort(unique(test.na.index))
    #     # los reemplazo con 0
    #     # testdataset[test.na.index,]<- 0
    #     # los elimino
    #     if (length(train.na.index)!=0){
    #       testdataset <- testdataset[-test.na.index,]
    #       testfilename <- testfilename[-test.na.index]
    #     }

    traindataset <- as.data.frame(data.matrix(traindataset))
    traindataset[,ncol(traindataset)] <- ifelse(traindataset[,ncol(traindataset)]==1,"YEMA","NOYEMA")

    testdataset <- as.data.frame(data.matrix(testdataset))
    testdataset[,ncol(testdataset)] <- ifelse(testdataset[,ncol(testdataset)]==1,"YEMA","NOYEMA")

    datasets[[i]]$ds <- traindataset
    datasets[[i]]$ts <- testdataset
    datasets[[i]]$prefijo <- datasets.pref[i]
    datasets[[i]]$testfilename <- testfilename
  }

  predictions <- list()
  for ( i in 1:length(test.data.files)) {
    prefijo <- datasets[[i]]$prefijo
    testData <- datasets[[i]]$ts

    results.c_svm <- NULL
    filename <- paste(dir_output,prefijo,"_csvm_bestmodel.RData",sep="")
    if (file.exists(filename)){
      load(filename)
      results.c_svm <- predict(c_svm_bestmodel,
                               newdata=testData[,-ncol(testData)],
                               probability = TRUE)
      results.c_svm.prob <- attr(results.c_svm, "probabilities")
      results.c_svm <- as.matrix(results.c_svm)
      results.c_svm <- factor(results.c_svm,levels=c("YEMA","NOYEMA"))
      pred.file <- paste(dir_output,prefijo,"_c_svm_predict.csv",sep="")
      x <- cbind(imageName=datasets[[i]]$testfilename, pred=as.character(results.c_svm),
                 probTrue=results.c_svm.prob[,1])
      cat(paste("Saving:",pred.file,"\n"))
      write.table(x,file=pred.file,sep=",")

      # load(filename)
      # results.c_svm <- predict(c_svm_bestmodel,
      #                          newdata=testData[,-ncol(testData)],
      #                          decision.values = TRUE)
      # results.c_svm <- factor(results.c_svm,levels=c("TRUE","FALSE"))
      # pred.file <- paste(dir_output,prefijo,"_c_svm_predict.csv",sep="")
      # x <- matrix(c(datasets[[i]]$testfilename,as.character(results.c_svm)),ncol=2,byrow=FALSE)
      # cat(paste("Saving:",pred.file,"\n"))
      # write.table(x,file=pred.file,sep=",")
    }else{
      results.c_svm <- NA
    }

    results.oc_svm <- NULL
    filename <- paste(dir_output,prefijo,"_ocsvm_bestmodel.RData",sep="")
    if (file.exists(filename)){
      load(filename)
      results.oc_svm <- predict(oc_svm_bestmodel,
                                newdata=testData[,-ncol(testData)])
      results.oc_svm <- factor(results.oc_svm,levels=c("YEMA","NOYEMA"))
      pred.file <- paste(dir_output,prefijo,"_oc_svm_predict.csv",sep="")
      x <- cbind(imageName=datasets[[i]]$testfilename, pred=as.character(results.oc_svm))
      cat(paste("Saving:",pred.file,"\n"))
      write.table(x,file=pred.file,sep=",")

      # load(filename)
      # results.oc_svm <- predict(oc_svm_bestmodel,
      #                           newdata=testData[,-ncol(testData)],
      #                           decision.values = TRUE)
      # results.oc_svm <- factor(results.oc_svm,levels=c("TRUE","FALSE"))
      # pred.file <- paste(dir_output,prefijo,"_oc_svm_predict.csv",sep="")
      # x <- matrix(c(datasets[[i]]$testfilename,as.character(results.oc_svm)),ncol=2,byrow=FALSE)
      # cat(paste("Saving:",pred.file,"\n"))
      # write.table(x,file=pred.file,sep=",")
    }else{
      results.oc_svm <- NA
    }

    predictions[[(i*2)-1]] <- results.c_svm
    predictions[[i*2]] <- results.oc_svm


  }

  truthvalues <- factor(testData[,ncol(testData)],levels=c("YEMA","NOYEMA"))

  ## RESULTS
  cat("\nRESULTS\n")
  results <- list(results=predictions,
                  algorithm.names=c("C-SVM"),
                  prefijos=datasets.pref,
                  true.values=truthvalues)

  generate.results(results)
}

## Funcion para correr el experimento
run.learning.curves <- function(){

  if(is.null(dir_input_train)) {
    stop("Falta settear directorios")
  }

  save <- FALSE
  dic_sizes <- seq(150,650,50)
  datapoints_sizes <- seq(690,10,-40)
  cvs <- 1:5
  outputLC.filename=paste(dir_output,"outputLC.csv",sep="")
  #inicializo con el header
  fileConn<-file(outputLC.filename)
  writeLines(paste(c("s,D,cv,alg,type",colnames(prf(1,1)$stats)),collapse=","), fileConn)
  close(fileConn)

  # appendToFile(colnames(prf(1,1)$stats),outputLC.filename)

  ## Read best_parameters
  best_parameters_file <- paste(dir_input_train,"best_parameters_binary.csv",sep="")
  best_parameters <- read.csv(best_parameters_file,
                              header=TRUE,
                              sep=",",
                              stringsAsFactors=FALSE)


  for (D in datapoints_sizes){
    for (s in dic_sizes){
      for (cv in cvs){

        # leo el dataset de datos train
        train.data.file <- paste(dir_input_train,"hist-s",s,"-D",D,"-cv",cv,".csv",sep="")
        mydataset <- read.csv(train.data.file,
                              header=TRUE,
                              sep=",",
                              stringsAsFactors=FALSE)

        colnames(mydataset)[ncol(mydataset)] <- "class"
        rownames(mydataset) <- 1:(nrow(mydataset))
        traindataset <- mydataset[,-c(1,2)] # elimino columnas sobrantes ID, nombre
        traindataset <- as.data.frame(data.matrix(traindataset))
        traindataset[,ncol(traindataset)] <- ifelse(traindataset[,ncol(traindataset)]==1,"YEMA","NOYEMA")
        trainfilename <- mydataset[,2]

        # leo el dataset de datos
        test.data.file <- paste(dir_input_test,"hist-s",s,"-D",D,"-cv",cv,".csv",sep="")
        mydataset <- read.csv(test.data.file,
                              header=TRUE,
                              sep=",",
                              stringsAsFactors=FALSE)

        #agrego class
        colnames(mydataset)[ncol(mydataset)] <- "class"
        rownames(mydataset) <- 1:(nrow(mydataset))

        testdataset <- mydataset[,-c(1,2)] # elimino columnas sobrantes ID, nombre
        testdataset <- as.data.frame(data.matrix(testdataset))
        testdataset[,ncol(testdataset)] <- ifelse(testdataset[,ncol(testdataset)]==1,"YEMA","NOYEMA")
        testfilename <- mydataset[,2]

        cat(paste("Train=",train.data.file,"\n",sep=""))
        # foreach(i=1:length(best_parameters) ) %dopar% {  # multicore
        for (i in 1:nrow(best_parameters)) {
          parameters <- best_parameters[i,]

          if (parameters[2]==s & parameters[3]=="c-svm"){
            ## C-SVM
            gamma <- parameters[4]
            cost <- parameters[5]
            type <- "C-classification"
            prefijo <- paste("s",s,"-D",D,"-cv",cv,sep="")
            filename <- paste(dir_output,prefijo,"_csvm_bestmodel.RData",sep="")

            if (save || !file.exists(filename)){
              try({c_svm_bestmodel <- svm(class~., data=traindataset,
                                          gamma = gamma,
                                          cost = cost,
                                          type=type,
                                          probability = TRUE)
                   save(c_svm_bestmodel,file=filename,compress="bzip2")
              })
            }else{
              load(filename)
            }
            #print(summary(c_svm_bestmodel))
            cat(".")

            try({
              results.train.c_svm <- predict(c_svm_bestmodel,
                                             newdata=traindataset[,-ncol(traindataset)],
                                             decision.values = TRUE)
              results.train.c_svm <- factor(results.train.c_svm,levels=c("YEMA","NOYEMA"))
            })

            try({
              results.test.c_svm <- predict(c_svm_bestmodel,
                                            newdata=testdataset[,-ncol(testdataset)],
                                            decision.values = TRUE)
              results.test.c_svm <- factor(results.test.c_svm,levels=c("YEMA","NOYEMA"))
            })

            try({
              output <- NA
              output <-(prf(results.train.c_svm,traindataset[,ncol(traindataset)]))
              salida <- paste(c(s,D,cv,"c-svm,train",output$stats),collapse=",")
              appendToFile(salida,outputLC.filename)
            })

            try({
              output <- NA
              output <-(prf(results.test.c_svm,testdataset[,ncol(testdataset)]))
              salida <- paste(c(s,D,cv,"c-svm,test",output$stats),collapse=",")
              appendToFile(salida,outputLC.filename)
            })

          }else if (parameters[2]==s & parameters[3]=="oc-svm"){
            gamma <- parameters[4]
            nu <- parameters[6]
            type <- "one-classification"
            prefijo <- paste("s",s,"-D",D,"-cv",cv,sep="")
            filename <- paste(dir_output,prefijo,"_ocsvm_bestmodel.RData",sep="")

            train_SY <- traindataset[which(traindataset[,'class']=="YEMA"),]

            if (save || !file.exists(filename)){
              try({oc_svm_bestmodel <- svm(class~., data=train_SY,
                                           gamma = gamma,
                                           nu=nu,
                                           type=type)
                   save(oc_svm_bestmodel,file=filename,compress="bzip2")
              })
            }else{
              load(filename)
            }
            #print(summary(oc_svm_bestmodel))
            cat(".")

            try({
              results.train.oc_svm <- predict(oc_svm_bestmodel,
                                              newdata=train_SY[,-ncol(train_SY)],
                                              decision.values = TRUE)
              results.train.oc_svm <- factor(results.train.oc_svm,levels=c("YEMA","NOYEMA"))
            })

            try({
              results.test.oc_svm <- predict(oc_svm_bestmodel,
                                             newdata=testdataset[,-ncol(testdataset)],
                                             decision.values = TRUE)
              results.test.oc_svm <- factor(results.test.oc_svm,levels=c("YEMA","NOYEMA"))
            })

            try({
              output <- NA
              output <-(prf(results.train.oc_svm,train_SY[,ncol(train_SY)]))
              salida <- paste(c(s,D,cv,"oc-svm,train",output$stats),collapse=",")
              appendToFile(salida,outputLC.filename)
            })

            try({
              output <- NA
              output <-(prf(results.test.oc_svm,testdataset[,ncol(testdataset)]))
              salida <- paste(c(s,D,cv,"oc-svm,test",output$stats),collapse=",")
              appendToFile(salida,outputLC.filename)
            })

          }
        }
      }
    }
  }
}

