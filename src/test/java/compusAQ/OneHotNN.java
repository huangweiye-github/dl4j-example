package compusAQ;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;

/**
 * TODO 替代所有文本处理为数据库处理
 * <p>
 * @author Scruel Tao
 */
public class OneHotNN {
  private MultiLayerNetwork model;

  public OneHotNN(int numInputs) throws Exception {
    initialModel(numInputs);
  }

  public static void main(String[] args) throws Exception {
    File charFile = new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\compusQA\\singleCharVec.txt");
    FileReader cfIn = new FileReader(charFile);
    BufferedReader bcfr = new BufferedReader(cfIn);
    LineNumberReader lnr = new LineNumberReader(bcfr);
    lnr.skip(charFile.length());
    new OneHotNN(lnr.getLineNumber());
  }

  public MultiLayerNetwork getModel() {
    return model;
  }

  public void initialModel(int numInputs) throws Exception {
    int seed = 123;
    File charFile = new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\compusQA\\answer.txt");
    FileReader cfIn = new FileReader(charFile);
    BufferedReader bcfr = new BufferedReader(cfIn);
    LineNumberReader lnr = new LineNumberReader(bcfr);
    // 定位到最后一行
    lnr.skip(charFile.length());
    // 获取结果集数量
    int numOutputs = lnr.getLineNumber() + 1;
    double learningRate = 0.1;
    int batchSize = 1000;
    int nEpochs = 500;
    int numHiddenNodes = 240;
//                int numHiddenNodes = numOutputs + numOutputs/4;

    // Load the training data:
    RecordReader rr = new CSVRecordReader();
//        rr.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_train.csv")));
    rr.initialize(new FileSplit(new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\compusQA\\w2v_tr.txt")));
    DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, numOutputs);

    // Load the test/evaluation data:
    RecordReader rrTest = new CSVRecordReader();
    rrTest.initialize(new FileSplit(new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\compusQA\\w2v_test.txt")));
    DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, numOutputs);

    // build neural network
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .weightInit(WeightInit.XAVIER)
        .updater(new Nesterovs(learningRate, 0.9))
//        .updater(new Sgd (learningRate))
        .activation(Activation.RELU)
        .list()
        .layer(0, new DenseLayer.Builder()
                .nIn(numInputs)
                .nOut(numHiddenNodes)
                .build())
        .layer(1, new DenseLayer.Builder()
                .nIn(numHiddenNodes)
                .nOut(numHiddenNodes)
                .build())
        .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(numHiddenNodes)
                .nOut(numOutputs)
                .build())
            .build();
    UIServer uiServer = UIServer.getInstance();
    StatsStorage statsStorage = new InMemoryStatsStorage();
    uiServer.attach(statsStorage);
    // 网络训练可视化
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(new ScoreIterationListener(1));
    model.setListeners(new StatsListener(statsStorage));
    // Initialize the user interface backend
      model.fit(trainIter,nEpochs);
    //Print the evaluation statistics
    System.out.println(model.evaluate(testIter).stats());
    this.model = model;
//    ModelSerializer.writeModel(model,new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model\\compusQA\\compusQA.bin"),false);
  }
}
