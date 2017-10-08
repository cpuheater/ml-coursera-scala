package com.cpuheater.ml

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._


object Ex1  extends App{

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)


  val numLinesToSkip = 0
  val delimiter = ","



  def computeCost(features: INDArray, labels: INDArray, theta: INDArray): Double = {
    val r = pow((features.mmul(theta.T)) - labels, 2)
    val r2 = r.sum(0)/(2*r.length)
    r2.getDouble(0)
  }


  def computeGradient(features: INDArray, labels: INDArray, theta: INDArray, alpha: Double, iters: Int): INDArray ={
    val temp = Nd4j.zerosLike(theta)
    val params = theta.length()
    val nbOfTrainingExamples = features.rows
    val updatedTheta = (0 to iters).foldLeft(temp)({
      case (accum, i) =>
        val error = features.mmul(accum.T) - labels
        (0 until params).map{
           p =>
            val r2 =  accum.getFloat(0, p) - (error * features.getColumn(p)).sum(0).mul(alpha/nbOfTrainingExamples).getFloat(0)
            accum.put(0, p, r2)
        }
        println(s"Cost: ${computeCost(features, labels, accum)}")
        accum
    })
    updatedTheta

  }

  val alpha = 0.01
  val iterations = 1500


  /**
    * Feature normalization - subtract mean, divide by standard deviation
    *
    **/

  def normalize(features: INDArray): INDArray = {
    val mean = features.mean(0)
    val std = features.std(0)
    (0 until features.columns()).foreach{
      col =>

        features(->, col) = (features(->, col) - mean.getColumn(col).getDouble(0)).div(std.getColumn(col).getDouble(0))
    }
    features
  }





  /**
    * y = theta1*x + theta0
    */

  def linearRegressionWithOneVariable(): Unit = {
    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("ex1/ex1data1.txt").getFile))


    val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,1,1, true)
    val dataSet: DataSet = iter.next()


    val features = dataSet.getFeatures()
    val labels = dataSet.getLabels()


    val bias = Nd4j.onesLike(features)
    val featuresWithBias =  Nd4j.concat(1, bias, features)



    val thetas =  Nd4j.create(Array(0d, 0d)).reshape(1, 2)



    val computedThetas = computeGradient(featuresWithBias, labels, thetas, alpha, iterations)
    println(s"theta0 = ${computedThetas.getColumn(0)} theta1=${computedThetas.getColumn(1)}")

  }


  def linearRegressionWithMultipleVariables(): Unit = {
    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("ex1/ex1data2.txt").getFile))


    val iter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000, 2, 2, true)
    val dataSet: DataSet = iter.next()

    val features = dataSet.getFeatures()
    val labels = dataSet.getLabels()

    val featuresNorm = normalize(features.dup())


    val featuresNormWithBias = Nd4j.concat(1, Nd4j.ones(featuresNorm.rows(), 1), featuresNorm)

    val thetas = Nd4j.zeros(1, featuresNormWithBias.columns())

    val computedThetas = computeGradient(featuresNormWithBias, labels, thetas, alpha, iterations)

    println(s"theta0 = ${computedThetas.getColumn(0)} theta1=${computedThetas.getColumn(1)} theta2=${computedThetas.getColumn(2)}")

  }




    linearRegressionWithOneVariable()

    linearRegressionWithMultipleVariables()








}
