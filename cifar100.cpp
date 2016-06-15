#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetCIFAR100.h"

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=50;
extern void printGPUMemoryUsage();

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
//  std::cout << "pic loadData start" << std::endl;
//  pic->loadData(1);
//  std::cout << "pic loadData finish" << std::endl;
  if (type==TRAINBATCH) {
    float
      c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
      c10=0, c11=1;
    if (rng.randint(2)==0) c00*=-1; //Horizontal flip
    c00*=1+rng.uniform(-0.2,0.2); // x stretch
    c11*=1+rng.uniform(-0.2,0.2); // y stretch
    int r=rng.randint(3);
    float alpha=rng.uniform(-0.2,0.2);
    if (r==0) matrixMul2x2inPlace(c00,c01,c10,c11,1,0,alpha,1); //Slant
    if (r==1) matrixMul2x2inPlace(c00,c01,c10,c11,1,alpha,0,1); //Slant other way
    if (r==2) matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha)); //Rotate
    pic->affineTransform(c00, c01, c10, c11);
    pic->jiggle(rng, 16);
    pic->colorDistortion(rng, 25.5, 0.15, 2.4, 2.4);
//    transformImage(pic->mat, backgroundColor, c00, c01, c10, c11);
//    pic->jiggle(rng,16);
//    std::cout << "distort matrix" << c00 << " " << c01 << std::endl << "\t\t" << c10 << " " << c11 << std::endl;
  }
  return pic;
}

float dropoutProbabilityMultiplier=0;// Set to 0.5 say to use dropout
int nFeaturesPerLevel(int i) {
  return 32*(i+1); //This can be increased
}


int main(int argc, char* argv[]) {
  std::string baseName="weights/cifar100";
  printGPUMemoryUsage();

  SpatiallySparseDataset trainSet=Cifar100TrainSet();
  SpatiallySparseDataset testSet=Cifar100TestSet();

  trainSet.summary();
  testSet.summary();
 
  printGPUMemoryUsage();
  ROFMPSparseConvNet cnn(
      2, 10, 32 /* 32n units in the n-th hidden layer*/, powf(2, 0.3333),
      VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses,
      0.1f /*dropout multiplier in the range [0,0.5] */, cudaDevice);

  if (argc > 1) {
    int epo = atoi(argv[1]);
    if (epo > 0) epoch = epo;
    else {
      for (int e = 10; e <= 230; e += 20) {
        cnn.loadWeights(baseName, e);
        cnn.processDatasetRepeatTest(testSet, batchSize / 2, 10);
      }
    }
  }
  printGPUMemoryUsage();
  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++; epoch <= 1000; epoch++) {
    printGPUMemoryUsage();
    std::cout << "epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.01 * epoch), 0.99);
    if (epoch % 10 == 0)
      cnn.saveWeights(baseName, epoch);
    if (epoch % 50 == 0)
      cnn.processDatasetRepeatTest(testSet, batchSize / 2, 10);
  }
  cnn.processDatasetRepeatTest(testSet, batchSize / 2, 100);
/*
  DeepCNet cnn(2,5,32,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);

  if (epoch>0)
    cnn.loadWeights(baseName,epoch);
  for (epoch++;;epoch++) {
    std::cout <<"epoch: " << epoch << std::flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.02 * epoch)); //reduce annealing rate for better results ...
    if (epoch%50==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(testSet,  batchSize);
    }
  }
*/
}
