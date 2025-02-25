package ocr;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class OcrTest {
    private static final long seed = 12345;
    private static final Random randNumGen = new Random(seed);
    public static void main(String[] args) throws IOException {
        // 2. 数据加载和预处理
        // 使用 RecordReader 加载图像数据
        ImageRecordReader traiRecordReader = new ImageRecordReader(1570, 720, 3, new ParentPathLabelGenerator()); // 可以设置初始的 height 和 width，但 DL4J 内部会处理不同大小的图像
        ImageTransform transform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "));
        traiRecordReader.initialize(new FileSplit(new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\ocr\\训练"), NativeImageLoader.ALLOWED_FORMATS, true),transform);
        DataSetIterator dataIterator = new RecordReaderDataSetIterator(traiRecordReader, 10, 1, traiRecordReader.numLabels());
        // DataNormalization 对图像数据进行归一化，使其位于 0 到 1 的范围内
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIterator);
        dataIterator.setPreProcessor(scaler);

        // 1. 定义网络结构
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(3) // 输入通道数，例如 RGB 图像为 3,屏幕截图和相机基本是RGB图像
                        .nOut(16)      // 输出通道数
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(16)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build()) // 进行全局平均池化，将卷积层的输出转换为固定大小的向量，从而可以处理任意大小的输入图像
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(32)
                        .nOut(traiRecordReader.numLabels()) // 类别数量
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.addListeners(new StatsListener(statsStorage, 5));
        model.addListeners(new ScoreIterationListener(5));
        // 3. 训练模型
        model.fit(dataIterator, 20);
        //测试
        ImageRecordReader testRecordReader = new ImageRecordReader(1570, 720, 3, new ParentPathLabelGenerator()); // 可以设置初始的 height 和 width，但 DL4J 内部会处理不同大小的图像
        testRecordReader.initialize(new FileSplit(new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\ocr\\测试"), NativeImageLoader.ALLOWED_FORMATS, true));
        DataSetIterator testDataIterator = new RecordReaderDataSetIterator(testRecordReader, 10, 1, testRecordReader.numLabels());
        Evaluation evaluate = model.evaluate(testDataIterator);
        System.out.println(evaluate.stats());
        System.out.println("MCC: "+evaluate.matthewsCorrelation(EvaluationAveraging.Macro));
    }

}
