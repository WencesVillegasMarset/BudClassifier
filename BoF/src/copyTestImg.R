

copy_yema_orig <- function(){
  dir.create("testImg",showWarnings=F)
  testImg <- read.csv("/home/ariel/IA/VIsE/vise/data/output/exp_2016-03-25_R/testImgList.txt",header=F,stringsAsFactors = F)
  infoPatch <- read.csv("/home/ariel/IA/VIsE/vise/data/images/corpus-26000/corpus-26000.csv",header=T,stringsAsFactors = F)
  testImg <- apply(testImg,2,basename)
  for ( c in 1:133){
    for (i in 1:nrow(infoPatch)){
      if ( testImg[c] == infoPatch[i,"imageName"] ) {
        imgOrig <- paste("/home/ariel/IA/VIsE/vise/data/images/yemas-orig/",infoPatch[i,"imageOrigin"],sep="")
        file.copy(imgOrig,"testImg",overwrite = F)
        break()
      }
    }
  }
}

copy_yema_mask <- function(){
  dir.create("testImg",showWarnings=F)
  testImg <- read.csv("/home/ariel/IA/VIsE/vise/data/output/exp_2016-03-25_R/testImgList.txt",header=F,stringsAsFactors = F)
  testImg <- apply(testImg,2,basename)
  for ( c in 1:133){
    patchBaseName <- sub("^([^.]*).*", "\\1", testImg[c])
    imgMaskFileName <- paste("/home/ariel/IA/VIsE/vise/data/images/corpus-26000/",patchBaseName,"_mask.png",sep="")
    file.copy(imgMaskFileName,"testImg",overwrite = F)


  }
}


