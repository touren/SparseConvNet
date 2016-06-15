#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetCIFAR10.h"

int epoch = 0;
int cudaDevice = -1; // PCI bus ID, -1 for default GPU
int batchSize = 50;

Picture *OpenCVPicture::distort(RNG &rng, batchType type) {
  OpenCVPicture *pic = new OpenCVPicture(*this);
//  std::cout << "matrix rows=" << pic->mat.rows << " cols=" << pic->mat.cols << " depth=" << pic->mat.depth() <<  std::endl;
  if (epoch <= 400 and type == TRAINBATCH) {
    // 2x2 identity matrix:
    // Generate an affine distortion matrix
    float c00 = 1, c01 = 0, c10 = 0, c11 = 1;
    c00 *= 1 + rng.uniform(-0.2, 0.2); // x stretch
    c11 *= 1 + rng.uniform(-0.2, 0.2); // y stretch
    if (rng.randint(2) == 0)           // Horizontal flip
      c00 *= -1;
    int r = rng.randint(3);
    float alpha = rng.uniform(-0.2, 0.2);
    if (r == 0) // Slant
      matrixMul2x2inPlace(c00, c01, c10, c11, 1, 0, alpha, 1);
    if (r == 1) // Slant
      matrixMul2x2inPlace(c00, c01, c10, c11, 1, alpha, 0, 1);
    if (r == 2) // Rotate
      matrixMul2x2inPlace(c00, c01, c10, c11, cos(alpha), -sin(alpha),
                          sin(alpha), cos(alpha));
    pic->affineTransform(c00, c01, c10, c11);
    pic->jiggle(rng, 16);
    pic->colorDistortion(rng, 25.5, 0.15, 2.4, 2.4);
    //std::cout << "distort matrix\t" << c00 << " " << c01 << std::endl << "\t\t" << c10 << " " << c11 << std::endl;
//    std::cout << "matrix rows=" << pic->mat.rows << " cols=" << pic->mat.cols << " depth=" << pic->mat.depth() <<  std::endl;
  }
  return pic;
}

int main(int argc, char* argv[]) {
  if (argc > 1) {
    int epo = atoi(argv[1]);
    if (epo > 0) epoch = epo;
  }

  std::string baseName = "weights/cifar10";

  SpatiallySparseDataset trainSet = Cifar10TrainSet();
  SpatiallySparseDataset testSet = Cifar10TestSet();

  trainSet.summary();
  testSet.summary();
  ROFMPSparseConvNet cnn(
      2, 12, 32 /* 32n units in the n-th hidden layer*/, powf(2, 0.3333),
      VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses,
      0.1f /*dropout multiplier in the range [0,0.5] */, cudaDevice);

  for (int e = 10; e <= 410; e += 30) {
    cnn.loadWeights(baseName, e);
    cnn.processDatasetRepeatTest(testSet, batchSize / 2, 20);
  }

  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++; epoch <= 410; epoch++) {
    std::cout << "epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.01 * epoch), 0.99);
    if (epoch % 10 == 0)
      cnn.saveWeights(baseName, epoch);
    if (epoch % 50 == 0)
      cnn.processDatasetRepeatTest(testSet, batchSize / 2, 15);
  }
  cnn.processDatasetRepeatTest(testSet, batchSize / 2, 100);
}
