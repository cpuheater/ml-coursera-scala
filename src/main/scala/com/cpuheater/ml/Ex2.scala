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
import org.nd4s.Evidences.float



object Ex2  extends App with Ex2Util{


  DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)


  val numLinesToSkip = 0
  val delimiter = ","


  /**
    * We are not going to use any optimization procedure but we will use
    * gradient descent to find parameters thetas
    */

  def logisticRegression(): Unit = {
    println("logisticRegression")
    val alpha = 0.001f
    val iterations = 90000

    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("ex2/ex2data1.txt").getFile))


    val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,2,2, true)
    val allData: DataSet = iter.next()


    val features = allData.getFeatures()
    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)
    val labels = allData.getLabels()

    val computedThetas = computeGradient(featuresWithBias, labels, alpha, iterations)

    println(s"Computed thetas ${computedThetas}")

    val positive = filterPositiveOrNegative(featuresWithBias, labels, 1)
    val negative = filterPositiveOrNegative(featuresWithBias, labels, 0)



    val positivePredicted  = Nd4j.getExecutioner().exec(new MatchCondition(hypothesis(positive, computedThetas), Conditions.greaterThan(0.5)),Integer.MAX_VALUE).getInt(0)

    val negativePredicted =  Nd4j.getExecutioner().exec(new MatchCondition(hypothesis(negative, computedThetas), Conditions.lessThan(0.5)),Integer.MAX_VALUE).getInt(0)

    println(s"Accuracy: ${(positivePredicted.toDouble + negativePredicted.toDouble)/featuresWithBias.rows()}")

  }



  def regularizedLogisticRegression(): Unit = {
    println("regularizedLogisticRegression")
    val alpha = 15f
    val iterations = 100000
    println("regularizedLogisticRegression")
    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("ex2/ex2data2.txt").getFile))
    val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,2,2, true)
    val allData: DataSet = iter.next()
    val features = allData.getFeatures()
    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)
    val labels = allData.getLabels()

    val featuresWithBiasMap = mapFeatures(featuresWithBias(->,1),featuresWithBias(->,2))

    val computedThetas = computeGradient(featuresWithBiasMap, labels, alpha, iterations, lambda = 1)


    val positive = filterPositiveOrNegative(featuresWithBiasMap, labels, 1)
    val negative = filterPositiveOrNegative(featuresWithBiasMap, labels, 0)



    val positivePredicted  = Nd4j.getExecutioner().exec(new MatchCondition(hypothesis(positive, computedThetas), Conditions.greaterThan(0.5)),Integer.MAX_VALUE).getInt(0)

    val negativePredicted =  Nd4j.getExecutioner().exec(new MatchCondition(hypothesis(negative, computedThetas), Conditions.lessThan(0.5)),Integer.MAX_VALUE).getInt(0)

    println(s"Accuracy: ${(positivePredicted.toDouble + negativePredicted.toDouble)/featuresWithBias.rows()}")



  }

  logisticRegression()

  regularizedLogisticRegression()


}


trait Ex2Util {

  def hypothesis(features: INDArray, thetas: INDArray) ={
    sigmoid(features.mmul(thetas.T))
  }

  def computeCost(features: INDArray, labels: INDArray, thetas: INDArray, lambda: Float = 0.0f): Float = {
    val output = hypothesis(features, thetas)
    val term1 = log(output).mul(-labels)
    val term2 = log(output.rsub(1)).mul(labels.rsub(1))
    Nd4j.clearNans(term2)
    val regularization =  (thetas(1 to term1.rows(), ->).mmul(thetas(1 to term1.rows(), ->).T) * (lambda/2)).getFloat(0)
    val crossEntropy =  (term1.sub(term2).sumNumber().floatValue() + regularization)/features.shape()(0)
    crossEntropy
  }


  def computeGradient(features: INDArray, labels: INDArray, alpha: Float, iters: Int, lambda: Float = 0.0f): INDArray ={
    val thetas =  Nd4j.zeros(features.columns(), 1).T
    val nbOfTrainingExamples = features.rows()
    val updatedTheta = (0 to iters).foldLeft(thetas)({
      case (thetas, i) =>
        val error = sigmoid(features.mmul(thetas.T)) - labels

        val grad = error.T.dot(features)  * alpha/nbOfTrainingExamples

        val regu = thetas(->, 1->) * lambda/nbOfTrainingExamples
        grad(->, 1->) = grad(->, 1->) + regu
        val updatedThetas =  thetas - grad
        println(s"Cost: ${computeCost(features, labels, updatedThetas)}")
        updatedThetas
    })
    updatedTheta

  }

  def filterPositiveOrNegative(features: INDArray, labels: INDArray, condition: Float): INDArray = {
    (0 until features.rows()).foldLeft(Option.empty[INDArray]){
      case (maybeINDArray, index) =>
        val label = labels.getRow(index).getColumn(0).getDouble(0)
        if(label == condition){
          val feature = features.getRow(index)
          if(maybeINDArray.isEmpty){
            Some(feature)
          }
          else {
            maybeINDArray.map {
              array =>
              Nd4j.concat(0, array, feature)
            }
          }
        }
        else
          maybeINDArray
    }.get
  }


  def mapFeatures(theta1: INDArray, theta2: INDArray): INDArray = {
    val degree = 6
    var out = Nd4j.ones(theta1.rows(), 1)
    (1 to  degree).map{
      index1 =>
        (0 to index1).map{
          index2 =>
            val r1 = pow(theta1, index1 - index2)
            val r2 = pow(theta2,  index2)
            val r3 = (r1 * r2).reshape( r1.rows(), 1 )
            out = Nd4j.hstack(out, r3)
        }
    }
    out
  }

}


