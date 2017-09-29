package com.cpuheater.ml

import com.cpuheater.util.Loader
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



object Ex3  extends App{


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
    val nbOfTrainingExamples = features.rows()
    val regularization =  (thetas(1 to term1.rows(), ->).mmul(thetas(1 to term1.rows(), ->)) * (lambda/2*nbOfTrainingExamples)).getDouble(0)
    val crossEntropy =  term1.sub(term2).sumNumber().doubleValue()/nbOfTrainingExamples + regularization
    crossEntropy
  }


  def computeGradient(features: INDArray, labels: INDArray, thetas: INDArray, alpha: Double, iters: Int, lambda: Double = 0.0): INDArray ={
    val temp = Nd4j.zerosLike(thetas)
    val params = thetas.length()
    val nbOfTrainingExamples = features.rows()
    val updatedTheta = (0 to iters).foldLeft(temp)({
      case (accum, i) =>
        val error = sigmoid(features.mmul(accum.T)) - labels
        (0 until params).map{
          p =>
            val regu = thetas(->, 1->).dot(thetas(->, 1->)).getDouble(0) * lambda/nbOfTrainingExamples
            val grad = alpha/nbOfTrainingExamples*(error * features.getColumn(p)).sum(0).getDouble(0) + regu
            val updatedTheta =  accum.getFloat(0, p) - grad
            accum.put(0, p, updatedTheta)
        }
        accum
    })
    updatedTheta

  }



  /**
    * We are not going to use ny optimization procedure but we will use
    * gradient descent to find parameters thetas
    */


  def logisticRegressionMultiClass(): Unit = {

    val content = Loader.load("ex3data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y").flatten)


    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)



    val thetas =Nd4j.create(featuresWithBias.columns(), 1)

    /*val computedThetas = computeGradient(featuresWithBias, labels, thetas, 0.001, 90000)


    println(hypothesis(Nd4j.create(Array(1.0, 45.0,85.0)), computedThetas))*/



  }




  def neuralNetwork() = {
    val content = Loader.load("ex3/ex3data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y").flatten)




    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithOnes =  Nd4j.concat(1, ones, features)


    val thetas = Loader.load("ex3/ex3weights.mat")

    val (theta1, theta2) = (Nd4j.create(thetas("Theta1")), Nd4j.create(thetas("Theta2")))


    def forwardPropagate(features: INDArray, theta1: INDArray, theta2: INDArray) : INDArray= {
      val z2 = features.mmul(theta1.T)
      val ones = Nd4j.ones(1)
      val a2 = Nd4j.concat(1, ones, sigmoid(z2))
      val z3 = a2.mmul(theta2.T)
      val output = sigmoid(z3)
      output
    }


    val correct = (0 until featuresWithOnes.rows()).foldLeft(0){
      case (accu, rowIndex) =>
        val dataRow = featuresWithOnes.getRow(rowIndex)
        val output = forwardPropagate(dataRow, theta1, theta2)
        val argMax = Nd4j.argMax(output, 1).getInt(0) +1
        val label = labels.getDouble(rowIndex).toInt
        if(argMax == label)
          accu + 1
        else
          accu

    }

    println(s"Neural network accuracy: ${correct/(featuresWithOnes.rows()toDouble)}")


  }

  neuralNetwork()


}


