package com.cpuheater.ml

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._



object Ex2  extends App{


  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)


  val numLinesToSkip = 0
  val delimiter = ","


  def hypothesis(features: INDArray, thetas: INDArray) ={
    val r = sigmoid(features.mmul(thetas.T))
    r
  }



  def computeCost(features: INDArray, labels: INDArray, thetas: INDArray, lambda: Double = 0.0): Double = {
    val output = hypothesis(features, thetas)
    val term1 = log(output).mul(-labels)
    val term2 = log(output.rsub(1)).mul(labels.rsub(1))
    Nd4j.clearNans(term2)
    val reguralization =  (thetas(1 to term1.rows(), ->).mmul(thetas(1 to term1.rows(), ->).T) * (lambda/2)).getDouble(0)
    val crossEntropy =  (term1.sub(term2).sumNumber().doubleValue() + reguralization)/features.shape()(0)
    crossEntropy
  }


  def computeGradient(features: INDArray, labels: INDArray, alpha: Double, iters: Int): INDArray ={
    val temp =  Nd4j.zeros(features.columns(), 1).T
    val params = temp.length()

    val updatedTheta = (0 to iters).foldLeft(temp)({
      case (accum, i) =>
        val error = sigmoid(features.mmul(accum.T)) - labels

        val r1 = error.T.dot(features)
        val updatedThetas =  accum - r1 * alpha/features.rows()
        println(s"Cost: ${computeCost(features, labels, updatedThetas)}")
        updatedThetas
    })
    updatedTheta

  }

  def filterPositiveOrNegative(features: INDArray, labels: INDArray, condition: Double): Array[Array[Double]] = {
    (0 until features.rows()).foldLeft(Array.empty[Array[Double]]){
      case (accum, index) =>
        val label = labels.getRow(index).getColumn(0).getDouble(0)
        if(label == condition){
          val feature = features.getRow(index)
          val row = Array(feature(0), feature(1), feature(2))
          accum:+row
        }
        else
          accum
    }
  }


  /**
    * We are not going to use ny optimization procedure but we will use
    * gradient descent to find parameters thetas
    */


  def logisticRegression(): Unit = {

    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("ex2/ex2data1.txt").getFile))


    val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,2,2, true)
    val allData: DataSet = iter.next()


    val features = allData.getFeatures()
    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)
    val labels = allData.getLabels()




    val computedThetas = computeGradient(featuresWithBias, labels, 0.001, 90000)

    val positive = Nd4j.create(filterPositiveOrNegative(featuresWithBias, labels, 1))
    val negative = Nd4j.create(filterPositiveOrNegative(featuresWithBias, labels, 0))



    val positivePredicted  = Nd4j.getExecutioner().exec(new MatchCondition(hypothesis(positive, computedThetas), Conditions.greaterThan(0.5)),Integer.MAX_VALUE).getInt(0)

    val negativePredicted =  Nd4j.getExecutioner().exec(new MatchCondition(hypothesis(negative, computedThetas), Conditions.lessThan(0.5)),Integer.MAX_VALUE).getInt(0)


    //println(hypothesis(Nd4j.create(Array(1.0, 45.0,85.0)), computedThetas))

    println(s"percentage positive examples correctly categorized ${positivePredicted.toDouble/positive.rows()}")
    println(s"percentage negative examples correctly categorized ${negativePredicted.toDouble/negative.rows()}")



  }



  def regularizedLogisticRegression(): Unit = {

    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("ex2/ex2data2.txt").getFile))


    val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,2,2, true)
    val allData: DataSet = iter.next()


    val features = allData.getFeatures()
    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)
    val labels = allData.getLabels()


    val computedThetas = computeGradient(featuresWithBias, labels, 0.001, 90000)

    println(s"${computedThetas}")


  }

  logisticRegression()

  //regularizedLogisticRegression()


}


