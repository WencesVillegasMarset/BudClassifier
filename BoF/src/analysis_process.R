options(error = recover)
require(compiler)
enableJIT(3)

library(gplots)
library(ggplot2)

library(doMC) # Multicore
cores <- 8 # Number of cores
registerDoMC(cores) # register cores


#########################################################################
#########################################################################
measures_process <- function() {
	dicts <- c(12,25,50,100,200,350,600)
	ratios <- c(1,2,4,8,16)

	dicts <- paste0("s",dicts)
	ratios <- paste0("r",ratios)

	dirOutput <- ""

	args <- commandArgs(TRUE)
	if (length(args)!=0) {
		dirOutput <- args[1]
	} else {
		dirOutput <- "analysis/measures/"
	}

	summaryFile <- paste0(dirOutput,"summary.csv")

	summary <- read.csv(summaryFile, sep=",", header=T)
	cat(summaryFile, " load\n")

	headerEst <- c( colnames(summary)[1:2], paste0("mean",colnames(summary)[4:10]), paste0("sd",colnames(summary)[4:10]) )

	fileEst <- paste0(dirOutput,"summary_est.csv")
	estatistics <- data.frame()

	if (file.exists(fileEst)) {
		estatistics <- read.csv(fileEst)
	} else {
		for (d in dicts) {
			for (r in ratios) {
				idx <- summary$Dict==d & summary$R==r
				meansD <- colMeans(summary[idx,4:10])
				sdsD <- apply(summary[idx,4:10], 2, sd)

				maxAccP <- max(summary[idx,4])
				minAccP <- min(summary[idx,4])
				maxPrecP <- max(summary[idx,5])
				minPrecP <- min(summary[idx,5])
				maxRecP <- max(summary[idx,6])
				minRecP <- min(summary[idx,6])
				maxFMP <- max(summary[idx,7])
				minFMP <- min(summary[idx,7])
			
				res <- matrix(c(d,r,meansD,sdsD,maxAccP,minAccP,maxPrecP,minPrecP,maxRecP,minRecP,maxFMP,minFMP), nrow=1, ncol=24)
				
				estatistics <- rbind(estatistics, res)
			}
		}

		colnames(estatistics) <- c(headerEst,"maxAccT","minAccT","maxPrecT","minPrecT","maxRecT","minRecT","maxFmT","minFmT")

		write.table(estatistics, file=fileEst, quote=F, sep=",", row.names=F)
	}

	sum_est <- estatistics

	#####################################################################
	#####################################################################
	fmTrue <- matrix(as.numeric(as.character(sum_est[,6])), nrow=5, ncol=7)
	colnames(fmTrue) <- unique(sum_est$Dict)
	rownames(fmTrue) <- unique(sum_est$R)

	fmTrueSD <- matrix(as.numeric(as.character(sum_est[,13])), nrow=5, ncol=7)
	colnames(fmTrueSD) <- unique(sum_est$Dict)
	rownames(fmTrueSD) <- unique(sum_est$R)

	#####################################################################
	#####################################################################
	# get the range for the x and y axis
	yrange <- c(.81, .95)#min(linesMtx), max(linesMtx))
	nlines <- nrow(fmTrue)

	filePlot <- paste0(dirOutput,"fmTrue-mean-bars.jpg")
	jpeg(filePlot,width=5,height=5,units='in',res=1000, quality=98)

	colnames(fmTrue) <- barsnames
	densbars <- seq(10,30,5)
	colorBars <- gray(seq(.15,.8,.16))

	# par(mar=c(5.1, 4.1, 4.1, 7.1), xpd=TRUE)
	par(mar=c(4.3, 2.7, 2.8, 0), xpd=TRUE)
	barCenters <- barplot(fmTrue, ylim=yrange, xpd=FALSE, xlab="Vocabulary Size (S)",
						  ylab="", yaxt="n", #density=densbars, angle=c(45,135,45,135,45),
			              col=colorBars, border=colorBars, width=2, beside=TRUE, space=c(0.3,2.5))

	lablist.y <- seq(round(yrange[1],2), round(yrange[2],2), by=0.02)
	axis(2, at=lablist.y, labels = FALSE)
	text(y=lablist.y, par("usr")[1], labels=lablist.y, srt=45, pos=2, offset=1, xpd=TRUE)

	legend("top",inset=c(0,-0.115), horiz=TRUE, #angle=c(45,135,45,135,45), density=densbars, 
		   fill=colorBars, border=colorBars, legend=linesnames)

	segments(barCenters, fmTrue - fmTrueSD * 2, barCenters,
	         fmTrue + fmTrueSD * 2, lwd = 1.5)

	arrows(barCenters, fmTrue - fmTrueSD * 2, barCenters,
	       fmTrue + fmTrueSD * 2, lwd = 1.5, angle = 90,
	       code = 3, length = 0.02)

	dev.off()

	fileCsv <- paste0(dirOutput,"fmTrue-mean-matrix.csv")
	write.table(fmTrue, file=fileCsv, quote=F, sep=",", row.names=F)

	fileCsv <- paste0(dirOutput,"fmTrueSD-mean-matrix.csv")
	write.table(fmTrueSD, file=fileCsv, quote=F, sep=",", row.names=F)

	#####################################################################
	#####################################################################
	rToPlot <- "r1"
	linesMtx <- rbind(	acc[rToPlot,],
						precTrue[rToPlot,],
						recTrue[rToPlot,],
						fmTrue[rToPlot,])
	rownames(linesMtx) <- c("accuracy","precision","recall","f-measure")

	sdMtx <- rbind(	accSD[rToPlot,],
					precSDTrue[rToPlot,],
					recSDTrue[rToPlot,],
					fmTrueSD[rToPlot,])
	rownames(sdMtx) <- c("accuracy","precision","recall","f-measure")

	idx <- sum_est[,2]==rToPlot
	maxMtx <- rbind(	sum_est[idx,17],
						sum_est[idx,19],
						sum_est[idx,21],
						sum_est[idx,23])
	rownames(maxMtx) <- c("accuracy","precision","recall","f-measure")

	minMtx <-  rbind(	sum_est[idx,18],
						sum_est[idx,20],
						sum_est[idx,22],
						sum_est[idx,24])
	rownames(minMtx) <- c("accuracy","precision","recall","f-measure")

	#####################################################################
	# get the range for the x and y axis
	xrange <- range(1:7)
	yrange <- range(.83, .98)#min(linesMtx), max(linesMtx))
	nlines <- nrow(linesMtx)

	filePlot <- paste0(dirOutput,"measures-lineplot-sd.jpg")
	# filePlot <- paste0(dirOutput,"measures-lineplot-maxmin.jpg")
	# jpeg(filePlot,width=1024,height=1024,pointsize=25, quality=100)
	jpeg(filePlot,width=5,height=5,units='in',res=1000, quality=98)

	par(mar=c(4.3, 2.7, 1, 1), xpd=TRUE)
	# set up the plot
	plot(xrange, yrange, type="n", xlab="Vocabulary Size (S)", ylab="", xaxt="n", yaxt="n")
	colors <- grey(c(0,.2,.4,.6)) #rainbow(nlines)
	linetype <- c(1:nlines)
	plotchar <- c(15:18)

	# add lines
	for (i in 1:nlines) {
		lines(1:length(dicts), linesMtx[i,], type="b", lwd=1.5, 
				lty=linetype[i], col=colors[i], pch=plotchar[i])
	  
	  	x <- 1:length(dicts)
	  	y <- linesMtx[i,]
	  	sd <- sdMtx[i,]
	  	# max <- maxMtx[i,]
	  	# min <- minMtx[i,]
		epsilon = 0.05
		for(j in 1:length(dicts)) {
			up = y[j] + sd[j]
			low = y[j] - sd[j]
			# up = max[j]
			# low = min[j]		
			segments(x[j], low , x[j], up, lwd=2, lty=linetype[i], col=colors[i])
			segments(x[j]-epsilon, up , x[j]+epsilon, up, lwd=1.5, col=colors[i])
			segments(x[j]-epsilon, low , x[j]+epsilon, low, lwd=1.5, col=colors[i])
		}
	  
	}

	lablist.y <- seq(round(yrange[1],2), round(yrange[2],2), by=0.02)

	axis(1, at=1:7, labels=c("12","25","50","100","200","350","600"))
	axis(2, at=lablist.y, labels = FALSE)
	text(y=lablist.y, par("usr")[1], labels=lablist.y, srt=45, pos=2, offset=1, xpd=TRUE)

	# add a title and subtitle
	#title("Accuracy, Precision, Recall and F-measure in S space")

	# add a legend
	legend(5.436, 0.986, rownames(linesMtx), cex=.9, lwd=1.5, col=colors,
	   pch=plotchar, lty=linetype)

	dev.off()

	fileCsv <- paste0(dirOutput,"measures-lineplot-sd.csv")
	write.table(round(linesMtx,3), file=fileCsv, quote=F, sep=",", row.names=T)

	fileCsv <- paste0(dirOutput,"measures-lineplot-mean.csv")
	write.table(round(sdMtx,3), file=fileCsv, quote=F, sep=",", row.names=T)


	##################################################################################
	##################################################################################
	linesMtx <- t(linesMtx)
	sdMtx <- t(sdMtx)

	idxlinesMax <- max.col(t(linesMtx),"first")
	idxSDMax <- max.col(t(1-sdMtx),"first")

	sNames <- rownames(linesMtx)
	sNames <- gsub("s","",sNames)

	tableLtx <- paste0(colnames(linesMtx), collapse=" & ")
	for (f in 1:nrow(linesMtx)) {

		idxMax <- idxlinesMax==f
		ltemp <- sprintf("%.3f",linesMtx[f,])
		if (any(idxMax))
			ltemp[idxMax] <- paste0("\\textbf{", ltemp[idxMax], "}")

		idxMax <- idxSDMax==f
		sdtemp <- sprintf("\\textit{(%.3f)}",sdMtx[f,])
		if (any(idxMax))
			sdtemp[idxMax] <- paste0("\\textbf{", sdtemp[idxMax], "}")

		ltemp <- c(paste0("\\multirow{2}{*}{",sNames[f],"}"), ltemp)
		sdtemp <- c(" ", sdtemp)
		tableLtx <- rbind(tableLtx, paste0(ltemp, collapse=" & "))
		tableLtx <- rbind(tableLtx, paste0(sdtemp, collapse=" & "))
		# tableLtx <- rbind(tableLtx, paste0(sdtemp, collapse=" & "))
	}

	tableLtx <- paste0(tableLtx[-1,], c(" \\\\", " \\\\ \\hline"))
	tableLtx <- c(paste0("$S$ & ", paste0(colnames(linesMtx), collapse=" & "), " \\\\ \\hline"), tableLtx)

	fileCsv <- paste0(dirOutput,"table-latex-mean-sd.txt")
	write.table(tableLtx, file=fileCsv, quote=F, sep="", row.names=F)

}


#########################################################################
#########################################################################
pert_process <- function(dirInput="", fileInfo="", dirOutput="", dict="", ratio="", iterations=0,
					nroSamples=1, overwrite=FALSE, onlyTestTrue=FALSE, plot=TRUE) {

	cat("\nProcesando diccionario", dict, "\n") 

	counts <- list()
	recall <- list()

	for (it in iterations) {
		fileCounts <- paste0(dirOutput,"count-matrix.csv")
		fileRecall <- paste0(dirOutput,"s",dict,"_r",ratio,"_it",it,"_recall-matrix.csv")

		if ( !file.exists(fileCounts) | !file.exists(fileRecall) | overwrite ) {
			#Cargo la info de las perturbaciones
			dataPertInfo <- read.csv(fileInfo, header=T, sep=",")
			origPatchUnique <- unique(dataPertInfo$origPatch)

			#Cargo las predicciones
			filePred <- paste0(dirInput,"s",dict,"_r",ratio,"_it",it,"_csvm_predict_pert.csv")
			dataPred <- read.csv(filePred, header=T, sep=",")

			dataPredInfo <- cbind(dataPertInfo,dataPred)

			#Solo procesar las imagenes de clase yema del testset clasificadas como true
			if (onlyTestTrue) {
				#TODO:
			}

			dataPert <- NULL
			if (nroSamples!=0) {
				rangos <- c(-1,10,20,30,40,50,60,70,80,90,100)
				for (opu in origPatchUnique) {
					idx <- dataPredInfo$origPatch == opu
					dataTemp <- dataPredInfo[idx,]

					dt <- NULL
					for (ry in 1:10) {
						for (rx in 1:10) {
							idx_rx <- dataTemp$yema_remain > rangos[rx] & dataTemp$yema_remain <= rangos[rx+1]
							idx_ry <- dataTemp$noyema_rel > rangos[ry] & dataTemp$noyema_rel <= rangos[ry+1]
							idx <- idx_rx & idx_ry
						
							if ( sum(idx)>=nroSamples ) {
								dt <- rbind(dt, dataTemp[idx,][sample(1:sum(idx), nroSamples),] )
							}
						}
					}
					dataPert <- rbind(dataPert, dt)
					cat(".")
				}
			} else {
				dataPert <- dataPred
			}

			#Separo entre perturbaciones TP y FN
			dataTrue <- dataPert[dataPert$pred=="YEMA",]
			dataFalse <- dataPert[dataPert$pred=="NOYEMA",]

			#Gráficos
			nbins <- 10
			hTrue <- hist2d(	x=dataTrue$yema_remain, 
								y=dataTrue$noyema_rel, 
								nbins=nbins, 
								show=F, 
								same.scale=T)

			hFalse <- hist2d(	x=dataFalse$yema_remain, 
								y=dataFalse$noyema_rel, 
								nbins=nbins, 
								show=F, 
								same.scale=T)

			colNamesMtx <- colnames(hTrue)

			mtxTRUE <- hTrue$counts[,10:1]
			mtxFALSE <- hFalse$counts[,10:1]

			colnames(mtxTRUE) <- colNamesMtx
			colnames(mtxFALSE) <- colNamesMtx

			counts[[it]] <- mtxTRUE + mtxFALSE
			recall[[it]] <- mtxTRUE / (mtxTRUE + mtxFALSE)

			if ( !file.exists(fileCounts) )
				write.table(x=counts[[it]], file=fileCounts, sep=",")
		
			write.table(x=recall[[it]], file=fileRecall, sep=",")

			cat(fileRecall," saved\n")

		} else {
			counts[[it]] <- as.matrix(read.csv(fileCounts))
			colnames(counts[[it]]) <- rownames(counts[[it]])#[10:1]
			recall[[it]] <- as.matrix(read.csv(fileRecall))
			colnames(recall[[it]]) <- rownames(recall[[it]])#[10:1]
			cat(fileRecall," loaded\n")
		}
	}

	recall_mean <- matrix(0, nrow=10, ncol=10)
	for (it in iterations) {
			recall_mean <- recall_mean + recall[[it]]
	}
	recall_mean <- recall_mean/length(iterations)

	idx_mtx <- counts[[iterations[1]]]<max(counts[[iterations[1]]])
	recall_mean[idx_mtx] <- NA

	if (plot) {
		filePlot <- paste0(dirOutput,"s",dict,"_r",ratio,"_mean-it_recall-matrix-win130k.jpg")
		# png(filePlot,width=800,height=800,pointsize=18)
		jpeg(filePlot,width=5,height=5,units='in',res=1000, quality=98)

		heatmap.2(recall_mean,scale="none",
					xlab="bud-pixels-relative",ylab="bud-pixels-kept",
					cexCol=1, cexRow=1,
					Colv=NA,Rowv=NA,
					dendrogram="none",
					key=FALSE,
					density.info="none",
					trace="none",			
					col=grey(seq(0.4,1,0.005)),
					cellnote=round(recall_mean,3),
					notecex=0.8,
                	notecol="black",
                	na.color="black",
                	srtCol=45, srtRow=45,
                	# lmat=matrix(c(1,1,1,1), 2, 2, byrow = TRUE),
                	lwid=c(1,5), lhei=c(1,5)
					)

		dev.off()

	}
}

process_2016.03.25_R_win130k <- function() {

	# Parámetros del experimento
	infoFile <- "../../data/images/perturbWindows130K/perturbInfo.csv"
	dirInput <- "analysis/pertub130k/"
	dicts <- c(12,25,50,100,200,350,600)
	ratios <- c(1)
	iterations <- 1:10
	nroSamples <- 4

	dirOutput <- paste0(dirInput,"nroSamples-",nroSamples,"/")

	if (!file.exists(dirOutput))
		dir.create(dirOutput)

	#foreach(i=1:length(dicts)) %dopar% { 
	for (i in 1:length(dicts)) {
		ds <- dicts[i]
		for (r in ratios) {
			res_process <- pert_process(dirInput=dirInput, fileInfo=infoFile, dirOutput=dirOutput, 
							dict=ds, ratio=r, iterations=iterations, nroSamples=nroSamples, 
							overwrite=FALSE, onlyTestTrue=FALSE, plot=TRUE) 
		}
	}

}


process_2016.03.25_R_BakGnd3000 <- function() {

	# Parámetros del experimento
	infoFile <- "../../data/images/TestCorpusBakGnd3000/output.csv"
	dir <- "../../output/exp_2016-03-25_R/TestCorpusBakGnd3000/"
	dirOutput <- "analysis/TestCorpusBakGnd/"

	dicts <- c(12,25,50,100,200,350,600)
	ratios <- 1
	iterations <- 1:10

	category <- c("B","C","F","I","N","Z","R","H","T","A","M","P")
	label <- c("Borde","Cercanias de la yema","Fuera de foco","Interno a troncos",
				"Nudos","Zarcillos","Racimos secos","Hojas secas","Tronco y corteza",
				"Alambre","Maleza","Postes")

	if (!file.exists(dirOutput))
		dir.create(dirOutput)

	dataPertInfo <- read.csv(infoFile, header=T, sep=",")

	results <- NULL
	results_mean <- NULL
	#foreach(i=1:length(dicts)) %dopar% { 
	for (ii in 1:length(dicts)) {
		ds <- dicts[ii]

		for (r in ratios) {
			for (it in iterations) {

				filePred <- paste0(dir, "s",ds,"_r",r,"_it",it,"_csvm_predict_pert.csv")
				dataPred <- read.csv(filePred, header=T, sep=",")

				data <- cbind(dataPertInfo, dataPred)

				for (i in 1:length(category)) {
					cat <- category[i]
					lab <- label[i]

					idxCat <- data$Type==cat
					dataTemp <- data[idxCat,]
					
					prec <- sum(dataTemp$pred=="NOYEMA")/nrow(dataTemp)

					results <- rbind(results, c(ds, r, it, cat, prec, lab))
				}	
			}

			for (i in 1:length(category)) {			
				idx <- results[,1]==ds & results[,2]==r & results[,4]==category[i]
				prec_mean <- mean(as.numeric(results[idx,5]))
				results_mean <- rbind(results_mean, c(ds, r, category[i], prec_mean, label[i]))
			}

		}

	}

	colnames(results) <- c("diccionary","ratio","it","category","precFalse","label")

	fileRes <- paste0(dirOutput, "results-precision-FALSE.csv")
	write.table(results,file=fileRes,sep=",", row.names=FALSE)

	colnames(results_mean) <- c("diccionary","ratio","category","precFalseMean","label")

	fileRes <- paste0(dirOutput, "results_mean-precision-FALSE.csv")
	write.table(results_mean,file=fileRes,sep=",", row.names=FALSE)



}
