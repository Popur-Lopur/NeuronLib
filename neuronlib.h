#ifndef NEURONLIB_H
#define NEURONLIB_H


#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QGuiApplication>
#include <QtQml>

#include <iostream>
#include <QObject>
#include <QVector>
#include <QRandomGenerator>
#include <QDebug>
#include <cmath>
#include <QFile>
#include <QTextStream>
#include <QtMath>
#include <QTime>
#include <QString>
#include <QLoggingCategory>
#include <QDateTime>
#include <QThread>


#include "neuronlib_global.h"


class NEURONLIBSHARED_EXPORT NeuronLib
{

public:
    NeuronLib();

    NeuronLib(int InputNum, int HiddenNum, int OutputNum, double LearningRate);




    int m_InputSize;    //количество входных нейронов
    int m_HiddenSize;   //количество скрытых нейронов
    int m_OutputSize;   //количество выходных нейронов

    double m_LearningRate; //коэф. обучения определяеться эксперементально

    QVector<double> m_InputNeuronsValues;   //значения входных нейронов
    QVector<double> m_HiddenNeuronsValues;  //значения скрытых нейронов
    QVector<double> m_OutputNeuronsValues;  //значения выходных нейронов

    QVector<double> m_TargetValue;
    QVector<double> m_DeltaHidden; //значения скрытых нейронов для ошибок
    QVector<double> m_DeltaOutput; //значения выходных нейронов для ошибок
    QVector<double> m_GradientErrorOutput; // градиент спуска выход (для обновления весов)
    QVector<double> m_GradientErrorHidden; // градиент спуска скрытый (для обновления весов)



    //двумерный потому что относяться веса одновременно к двум слоям
    QVector<QVector<double>> m_InputBetweenHidden_Weights;  //веса между входным и скрытым
    QVector<QVector<double>> m_HiddenBetweenOutput_Weights; //веса между скрытым и входным

    QVector<double> m_HiddenBiases; //смещение скрытых нейронов
    QVector<double> m_OutputBiases; //смещение выходных нейронов

    double SigmoidDerivative(double x); // производная функция Sigmoid
    double Sigmoid(double x);  //функция активации Sigmoid

    QVector<double> m_OutputErrorValues; //значения ошибок на выходном слое при обратном распространении
    QVector<double> m_HiddenErrorValues; //значения ошибок на скрытом слое при обратном распространении


    void MinMax(QVector<double> & _input);
    void FeedForward( const QVector<double>& _input); //функция прямого распространения

    struct CsvLine
    {
        QString name_kp_in_csvline;
        QString  id_in_csvline;
        QVector<double> values_in_csvline;
    };

    struct CsvLineTrain
    {
        QVector<double> target_out_in_csvline;
        QString name_kp_in_csvline;
        QString  id_in_csvline;
        QVector<double> values_in_csvline;
    };

    void GetDataFromFile(QString filename);
    QList<CsvLine> m_data_list;
    QList<CsvLineTrain> m_data_train_list;
    void  LoadDataWeights (QString filename, NeuronLib& nn);
    void LoadDataStruct (QString filename,int& inp, int& hid, int& out, double& lr);








    void neuronTest();

};

#endif // NEURONLIB_H
