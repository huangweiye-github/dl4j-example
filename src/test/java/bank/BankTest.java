package bank;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.inmemory.InMemoryRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.StringSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

/***
 * 预测银行客户是否流失
 */

public class BankTest {

    private static TransformProcess transformProcess;
    private static MultiLayerNetwork model;
    private static String[] exitedCategoricalValue = {"0", "1"};//结果分类
    private static Schema finalSchema;
    public static void main(String[] args) throws Exception {
        Random random = new Random();
        random.setSeed(0xC0FFEE);

        FileSplit inputSplit = new FileSplit(new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\bank\\训练"), random);
        CSVRecordReader recordReader = new CSVRecordReader(1);
//        FileRecordReader recordReader = new FileRecordReader();
        recordReader.initialize(inputSplit);
        Schema schema = new Schema.Builder()
                .addColumnsInteger("Row Number", "Customer Id")
                .addColumnString("Surname")
                .addColumnInteger("Credit Score")
                .addColumnCategorical("Geography", "France", "Germany", "Spain")
                .addColumnCategorical("Gender", "Female", "Male")
                .addColumnsInteger("Age", "Tenure")
                .addColumnDouble("Balance")
                .addColumnInteger("Num Of Products")
                .addColumnCategorical("Has Credit Card", "0", "1")
                .addColumnCategorical("Is Active Member", "0", "1")
                .addColumnDouble("Estimated Salary")
                .addColumnCategorical("Exited", exitedCategoricalValue)
                .build();
        DataAnalysis analysis = AnalyzeLocal.analyze(schema, recordReader);
        HtmlAnalysis.createHtmlAnalysisFile(analysis, new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model\\bank\\analysis.html"));
        transformProcess = new TransformProcess.Builder(schema)
                /**移除无意义的列*/
                .removeColumns("Row Number", "Customer Id", "Surname")
                /**单热列(独热)
                 * 当类别数据被简单地编码为整数时（例如，红色=1，蓝色=2，绿色=3），
                 * 模型可能会错误地学习到类别之间存在某种排序关系。
                 * 独热表示消除了这种潜在的误解
                 * 独热表示通常可以帮助模型更好地区分不同的类别。
                 * 例如：
                 * 如果 "Geography" 列有三个类别（"France", "Germany", "Spain"），
                 * 那么它将被转换为三列："Geography_France", "Geography_Germany", "Geography_Spain"，
                 * 每一列的值是 0 或 1
                 * */
                .categoricalToOneHot("Geography", "Gender", "Has Credit Card", "Is Active Member")
                /**
                 * 独热编码，但只针对 1 到 4 的值,超出范围，则可能被忽略或者以其他方式处理（具体取决于 DataVec 库的实现）
                 */
                .integerToOneHot("Num Of Products", 1, 4)
                /**
                 * 不同的特征可能具有完全不同的取值范围。
                 * 例如，一个特征的取值可能在 0 到 1 之间，而另一个特征的取值可能在 1000 到 100000 之间
                 * 影响： 这会导致一些问题：
                 *      距离计算偏差： 许多机器学习算法（例如 K 近邻、支持向量机）依赖于距离计算。如果特征的尺度差异很大，那么取值范围大的特征会对距离计算产生更大的影响，导致模型偏向于这些特征。
                 *      梯度下降问题： 在使用梯度下降算法训练模型时（例如神经网络），如果特征的尺度差异很大，会导致梯度在不同方向上的变化幅度不同，从而导致收敛速度慢，甚至无法收敛。
                 * 作用： 归一化将所有特征的取值范围缩放到一个相同的尺度（例如 0 到 1），从而消除特征尺度差异带来的影响，使模型能够公平地对待所有特征。
                 **/
                .normalize("Tenure", Normalize.MinMax, analysis)
                .normalize("Age", Normalize.Standardize, analysis)
                .normalize("Credit Score", Normalize.Log2Mean, analysis)
                .normalize("Balance", Normalize.Log2MeanExcludingMin, analysis)
                .normalize("Estimated Salary", Normalize.Log2MeanExcludingMin, analysis)
                .build();
        finalSchema = transformProcess.getFinalSchema();


        TransformProcessRecordReader trainRecordReader = new TransformProcessRecordReader(new CSVRecordReader(1), transformProcess);
        trainRecordReader.initialize(inputSplit);
        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator.Builder(trainRecordReader, 1000)
                /**输出结果分类*/
                .classification(finalSchema.getIndexOfColumn("Exited"), 2)
                .build();
        model = new MultiLayerNetwork(getMultiLayerConfiguration());
        model.init();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.addListeners(new StatsListener(statsStorage, 50));
        model.addListeners(new ScoreIterationListener(50));
        //训练
        model.fit(trainIterator, 500);
        //测试
        TransformProcessRecordReader testRecordReader = new TransformProcessRecordReader(new CSVRecordReader(1), transformProcess);
        testRecordReader.initialize( new FileSplit(new File("D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\bank\\测试")));
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator.Builder(testRecordReader, 1000)
                .classification(finalSchema.getIndexOfColumn("Exited"), 2)
                .build();
        Evaluation evaluate = model.evaluate(testIterator);
        System.out.println(evaluate.stats());
        System.out.println("MCC: "+evaluate.matthewsCorrelation(EvaluationAveraging.Macro));
        //验证
        valid("1,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1");
    }

    private static MultiLayerConfiguration getMultiLayerConfiguration() {
/*
        ========================Evaluation Metrics========================
         # of classes:    2
        Accuracy:        0.8717
        Precision:       0.7964
        Recall:          0.4973
        F1 Score:        0.6123
        Precision, recall & F1: reported for positive class (class 1 - "1") only
*/
//
//        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
//                .seed(0xC0FFEE)
//                .weightInit(WeightInit.XAVIER)
//                .activation(Activation.TANH)
//                .updater(new Adam.Builder().learningRate(0.001).build())
//                .l2(0.000316)
//                .list(
//                        new DenseLayer.Builder().nOut(25).build(),
//                        new DenseLayer.Builder().nOut(25).build(),
//                        new DenseLayer.Builder().nOut(25).build(),
//                        new DenseLayer.Builder().nOut(25).build(),
//                        new DenseLayer.Builder().nOut(25).build(),
//                        new OutputLayer.Builder(new LossMCXENT()).nOut(2).activation(Activation.SOFTMAX).build()
//                )
//                .setInputType(InputType.feedForward(finalSchema.numColumns() - 1))
//                .build();
//        return config;

/*
========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.9195
 Precision:       0.7588
 Recall:          0.8866
 F1 Score:        0.8177
Precision, recall & F1: reported for positive class (class 1 - "1") only

*/

        double learningRate = 0.1;
        // 获取结果集数量
        int numOutputs = exitedCategoricalValue.length;
        int numHiddenNodes = 240;
        int numInputs = finalSchema.numColumns() - 1;
        int seed = 123;
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
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
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .build())
                .build();
        return config;
    }

    private static void valid(String validCvsRowData) throws IOException, InterruptedException {
        Scanner input = new Scanner(System.in);
        String text;
        while (true) {
            try {
                validCvsRowData = input.nextLine();
                TransformProcessRecordReader validRecordReader = new TransformProcessRecordReader(new CSVRecordReader(0), transformProcess);
                /**如果用的是 CSVRecordReader ，则 StringSplit 中的数据要符合 CSVRecordReader 的格式*/
                validRecordReader.initialize(new StringSplit(validCvsRowData));
                RecordReaderDataSetIterator validIterator = new RecordReaderDataSetIterator.Builder(validRecordReader, 1000)
                        .classification(finalSchema.getIndexOfColumn("Exited"), 2)
                        .build();
                INDArray predicted = model.output(validIterator, false);
                System.out.println(predicted);
                INDArray binaryGuesses = predicted.gt(0.5).castTo(DataType.FLOAT);
                //只有两种分类
                for (int i = 0; i < exitedCategoricalValue.length; i++) {
                    if (binaryGuesses.getInt(i) == 1){
                        System.out.println("分类："+binaryGuesses.getInt(i));
                    }
                }
            }catch (Exception exception){
                exception.printStackTrace();
            }
        }

    }


}
