#include "neuronlib.h"

NeuronLib::NeuronLib()
{
}

NeuronLib::NeuronLib(int InputNum, int HiddenNum, int OutputNum, double LearningRate)
{
    m_InputSize = InputNum;
    m_HiddenSize = HiddenNum;
    m_OutputSize = OutputNum;
    m_LearningRate = LearningRate;
    m_InputNeuronsValues.resize(InputNum);
    m_HiddenNeuronsValues.resize(HiddenNum);
    m_OutputNeuronsValues.resize(OutputNum);
    m_DeltaHidden.resize(HiddenNum);
    m_DeltaOutput.resize(OutputNum);
    m_TargetValue.resize(OutputNum);
    m_OutputErrorValues.resize(OutputNum);
    m_HiddenErrorValues.resize(HiddenNum);
    m_GradientErrorOutput.resize(OutputNum);
    m_GradientErrorHidden.resize(HiddenNum);

    std::srand(std::time(nullptr));

    //***ЗАПОЛНЕНИЕ ВЕСОВ МЕЖДУ ВХОДНЫМ И СКРЫТЫМ СЛОЯМИ***

    // делаем размер вектора<вектора> весов(между входным и скрытым слоями) равным количеству входных нейронов (грубо веса стоят в точке входа и этот цикл выясняет сколько этих точек входа для весов)
    m_InputBetweenHidden_Weights.resize(m_InputSize);
    for (int i = 0; i < m_InputSize; ++i)
    {
        // делаем размер  i - вектора весов в векторе-векторе равным количеству скрытых нейронов (количество связей 1 нейрона на входе !входного! слоя с общим количеством нейронов на скрытом слое)
        m_InputBetweenHidden_Weights[i].resize(m_HiddenSize);
        for (int j = 0; j < m_HiddenSize; ++j)
        {
            double randomValue =  -0.5 + static_cast<double>(std::rand()) / RAND_MAX * (0.5 - (-0.5));
            m_InputBetweenHidden_Weights[i][j] = randomValue;
            //qDebug() << m_InputBetweenHidden_Weights[i][j];
        }

    }

    //***ЗАПОЛНЕНИЕ ВЕСОВ МЕЖДУ СКРЫТЫМ И ВЫХОДНЫМ СЛОЯМИ***

    // делаем размер вектора<вектора> весов(между скрытым и выходным слоями) равным количеству скрытых нейронов (грубо веса стоят в точке входа  !на скрытном! слое и этот цикл выясняет сколько этих точек входа для весов)
    m_HiddenBetweenOutput_Weights.resize(m_HiddenSize);
    for (int i = 0; i < m_HiddenSize; ++i)
    {
        // делаем размер  i - вектора весов в векторе-векторе равным количеству выходных нейронов (количество связей 1 нейрона на входе !в скрытном! слое с общим количеством нейронов на выходном слое)
        m_HiddenBetweenOutput_Weights[i].resize(m_OutputSize);
        for (int j = 0; j < m_OutputSize; ++j)
        {
            double randomValue = -0.5 + static_cast<double>(std::rand()) / RAND_MAX * (0.5 - (-0.5));
            m_HiddenBetweenOutput_Weights[i][j] = randomValue;
            //qDebug() << m_HiddenBetweenOutput_Weights[i][j];
        }


    }

    //***ЗАПОЛНЕНИЕ СМЕЩЕНИЙ СКРЫТОГО СЛОЯ***
    m_HiddenBiases.resize(m_HiddenSize);
    for (int i = 0; i < m_HiddenSize; ++i)
    {
        double randomValue = -0.5 + static_cast<double>(std::rand()) / RAND_MAX * (0.5 - (-0.5));
        m_HiddenBiases[i] = randomValue;
        //qDebug() << m_HiddenBiases[i];
    }

    //***ЗАПОЛНЕНИЕ СМЕЩЕНИЙ ВЫХОДНОГО СЛОЯ***
    m_OutputBiases.resize(m_OutputSize);
    for (int i = 0; i < m_OutputSize; ++i)
    {
        double randomValue = -0.5 + static_cast<double>(std::rand()) / RAND_MAX * (0.5 - (-0.5));
        m_OutputBiases[i] = randomValue;
        //qDebug() << m_OutputBiases[i];
    }

}

double  NeuronLib::Sigmoid(double x)
//Сигмоида
{
    return 1 / (1 + exp(-x));
}

double  NeuronLib::SigmoidDerivative(double x)
//Производная сигмоиды
{
    return x * (1 - x);

}
void  NeuronLib::MinMax(QVector<double>& _input) {
    //Нормализация

    double min = *std::min_element(_input.constBegin(), _input.constEnd());
    double max = *std::max_element(_input.constBegin(), _input.constEnd());

    for (int i = 0; i < _input.size(); ++i)
    {
        _input[i] = (_input[i] - min) / (max - min);
    }



}
void  NeuronLib::FeedForward( const QVector<double>& _input)
{
    m_InputNeuronsValues = _input;

    // Расчет значений скрытых нейронов
    for (int i = 0; i < m_HiddenSize; ++i)
    {
        double activation = m_HiddenBiases[i];
        for (int j = 0; j < m_InputSize; ++j)
        {
            // сумма всех Нитей(нейрон входа * вес) идущих к нейрону на скрытом слое
            activation += m_InputNeuronsValues[j] * m_InputBetweenHidden_Weights[j][i];
        }
        // получение значений нейронов на скрытом слое
        m_HiddenNeuronsValues[i] = Sigmoid(activation);

    }


    // Расчет значений выходных нейронов

    for (int i = 0; i < m_OutputSize; ++i)
    {
        double activation = m_OutputBiases[i];
        for (int j = 0; j < m_HiddenSize; ++j)
        {
            // сумма всех Нитей(нейрон скрытый * вес) идущих к нейрону на выходном слое
            activation += m_HiddenNeuronsValues[j] * m_HiddenBetweenOutput_Weights[j][i];
        }
        // получение значений нейронов на выходном слое
        m_OutputNeuronsValues[i] = Sigmoid(activation);
    }

}


void NeuronLib::GetDataFromFile(QString filename)
{
    QFile file(filename);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream stream(&file);
        while (!stream.atEnd())
        {
            QString line = stream.readLine();
            QStringList values = line.split(",");
            QString name_of_kp = values[0];
            QString id = values[1];
            QVector<double> vec;

            for (int i = 2; i < values.size(); ++i)
            {
                double value = values[i].toDouble();
                vec.push_back(value);
            }

            m_data_list.push_back({name_of_kp, id, vec});
        }
        file.close();
    }
    else {
        qDebug() << "Failed to open file";
        return ;
    }
}

void NeuronLib::GetTrainDataFromFile(QString filename_train)
{
    QFile file(filename_train);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream stream(&file);
        while (!stream.atEnd())
        {
            QString line = stream.readLine();
            QStringList values = line.split(",");
            double target = values[0].toDouble();
            QVector<double> target_out;
            target_out.push_back(target);
            QString name_of_kp = values[1];
            QString id = values[2];

            QVector<double> vec;

            for (int i = 3; i < values.size(); ++i)
            {
                double value = values[i].toDouble();
                vec.push_back(value);
            }

            m_data_train_list.push_back({target_out, name_of_kp, id, vec});
        }

        file.close();
    }
    else {
        qDebug() << "Failed to open file";
        return ;
    }
}



void NeuronLib::SaveDataWeights(QString filename, NeuronLib& nn)
{

    QFile file(filename);
    QTextStream out(&file);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        for (int i = 0; i < nn.m_InputSize; ++i)
        {
            out  << "Input-Hidden" ;
            for (int j = 0; j < nn.m_HiddenSize; ++j)
            {
                out  << " " << nn.m_InputBetweenHidden_Weights[i][j];
            }
            out << endl;
        }

        for (int i = 0; i < nn.m_HiddenSize; ++i)
        {
            out <<  "Hidden-Output" ;
            for (int j = 0; j < nn.m_OutputSize; ++j)
            {
                out  << " " << nn.m_HiddenBetweenOutput_Weights[i][j];
            }
            out << endl;
        }

        for (int i = 0; i < nn.m_HiddenSize; ++i)
        {
            out << "Hidden Biases" << " " << nn.m_HiddenBiases[i];
        out << endl;
        }

        for (int i = 0; i < nn.m_OutputSize; ++i)
        {
            out << "Output Biases" << " " << nn.m_OutputBiases[i];
        out << endl;
        }
        file.close();
        qDebug() << "Weights successful saved in:" << filename << endl;
    }
    else {
        qDebug() << "Weights  didn't  saved in:" << filename << endl;
    }
}

void NeuronLib::LoadDataWeights(QString filename, NeuronLib& nn)
{
    QVector<QString> IHvec;
    QVector<QString> HOvec;
    QVector<QString> HBvec;
    QVector<QString> OBvec;

    QFile file(filename);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&file);

        while (!in.atEnd()) {
            QString line = in.readLine();
            if (line.startsWith("Input-Hidden"))
            {
                line.remove("Input-Hidden ");
                IHvec.append(line);
            }
            if (line.startsWith("Hidden-Output"))
            {
                line.remove("Hidden-Output ");
                HOvec.append(line);
            }
            if (line.startsWith("Hidden Biases"))
            {
                line.remove("Hidden Biases ");
                HBvec.append(line);
            }
            if (line.startsWith("Output Biases"))
            {
                line.remove("Output Biases ");
                OBvec.append(line);
            }
        }

        file.close();

        for (int i = 0; i < IHvec.size(); ++i)
        {
            QStringList values = IHvec[i].split(" ");
            for (int j = 0; j < values.size(); ++j) {
                nn.m_InputBetweenHidden_Weights[i][j] = values[j].toDouble();
            }
        }

        for (int i = 0; i < HOvec.size(); ++i)
        {
            QStringList values = HOvec[i].split(" ");
            for (int j = 0; j < values.size(); ++j) {
                nn.m_HiddenBetweenOutput_Weights[i][j] = values[j].toDouble();
            }
        }

        for (int i = 0; i < HBvec.size(); ++i)
        {
            QStringList values = HBvec[i].split(" ");
            for (int j = 0; j < values.size(); ++j) {
                nn.m_HiddenBiases[i] = values[j].toDouble();
            }
        }

        for (int i = 0; i < OBvec.size(); ++i)
        {
            QStringList values = OBvec[i].split(" ");
            for (int j = 0; j < values.size(); ++j) {
                nn.m_OutputBiases[i] = values[j].toDouble();
            }
        }

        qDebug() << "Weights successful downloaded in:" << filename << endl;
    }

    else {
        qDebug() << "Weights  didn't  downloaded in:" << filename << endl;
    }
}




void NeuronLib::neuronTest()
{
    NeuronLib* data_test_csv = new NeuronLib();
    data_test_csv->GetDataFromFile("C:\\Work\\QvaziWindow\\test.csv");

    int NumberInputs = data_test_csv->m_data_list.at(0).values_in_csvline.size();
    NeuronLib Net =  NeuronLib(NumberInputs,400,1,0.003);

    NeuronLib data = NeuronLib();
    data.LoadDataWeights("C:\\Work\\Qvazi\\weights.txt", Net);

    for (int i = 0; i < data_test_csv->m_data_list.size(); ++i) {

        QVector<double> inputs =  data_test_csv->m_data_list.at(i).values_in_csvline;
        Net.MinMax(inputs);
        Net.FeedForward(inputs);

        qDebug() << "Number of line: "  << i + 1 ;
        qDebug() << "Name  KP: " << data_test_csv->m_data_list.at(i).name_kp_in_csvline;
        qDebug() << "ID: " << data_test_csv->m_data_list.at(i).id_in_csvline ;

        for (int j = 0; j < Net.m_OutputSize; ++j) {
            qDebug() << "Predict:" << Net.m_OutputNeuronsValues[j];
        }
        qDebug() << "------------------";
    }
}

