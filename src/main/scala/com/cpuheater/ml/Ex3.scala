package com.cpuheater.ml

import com.cpuheater.util.Loader
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4s.Evidences.float



object Ex3  extends App with Ex3Util{


  DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)


  val numLinesToSkip = 0
  val delimiter = ","

  /**
    * We are not going to use ny optimization procedure but we will use
    * gradient descent to find parameters thetas
    */


  def logisticRegressionMultiClass(): Unit = {

    val alpha = 4f
    val iters = 100

    val content = Loader.load("ex3/ex3data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y"))


    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)


    val allThetas = computeThetasForEachClass(featuresWithBias, labels, alpha, iters)


    def classPrediction(features: INDArray, thetas: INDArray): Float = {
     val predictions =  (0 until 10).map{
        index =>
          val pred =  hypothesis(features, thetas(index, ->))
          pred.getFloat(0)
      }
      val max = Nd4j.argMax(Nd4j.create(predictions.toArray)).getFloat(0)

      if(max == 0) 10.0f else max
    }

    val (total, correct) = (0 until featuresWithBias.rows()).foldLeft((0, 0)){
      case ((total, correct), index) =>
        val pred = classPrediction(featuresWithBias.getRow(index), allThetas)
        if(classPrediction(featuresWithBias.getRow(index), allThetas) == labels.getRow(index).getFloat(0))
          (total+1, correct+1)
        else
          (total+1, correct)
    }

    println(s"Logistic Regression Multi Class Accuracy ${correct.toDouble/total}")

  }




  def neuralNetwork() = {
    val content = Loader.load("ex3/ex3data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y").flatten)

    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)

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


    val correct = (0 until featuresWithBias.rows()).foldLeft(0){
      case (accu, rowIndex) =>
        val dataRow = featuresWithBias.getRow(rowIndex)
        val output = forwardPropagate(dataRow, theta1, theta2)
        val argMax = Nd4j.argMax(output, 1).getInt(0) +1
        val label = labels.getDouble(rowIndex).toInt
        if(argMax == label)
          accu + 1
        else
          accu

    }

    println(s"Neural network accuracy: ${correct/(featuresWithBias.rows()toDouble)}")


  }

  neuralNetwork()


  logisticRegressionMultiClass()


}


trait Ex3Util {


  def hypothesis(features: INDArray, thetas: INDArray) ={
    sigmoid(features.mmul(thetas.T))
  }

  def computeCost(features: INDArray, labels: INDArray, thetas: INDArray, lambda: Float = 0.0f): Float = {
    val output = hypothesis(features, thetas)
    val term1 = log(output).mul(-labels)
    val term2 = log(output.rsub(1)).mul(labels.rsub(1))
    Nd4j.clearNans(term2)
    val nbOfTrainingExamples = features.rows()
    val regularization =  (thetas(1 to term1.rows(), ->).mmul(thetas(1 to term1.rows(), ->).T) * (lambda/2*nbOfTrainingExamples)).getFloat(0)
    val crossEntropy =  term1.sub(term2).sumNumber().floatValue()/nbOfTrainingExamples + regularization
    crossEntropy
  }


  def computeGradient(features: INDArray, labels: INDArray, alpha: Float, iters: Int, lambda: Float = 0.0f): INDArray ={
    val thetas =  Nd4j.zeros(features.columns(), 1).T
    val nbOfTrainingExamples = features.rows()
    val updatedTheta = (0 to iters).foldLeft(thetas)({
      case (thetas, i) =>
        val error = sigmoid(features.mmul(thetas.T)) - labels
        val regu = thetas(->, 1->) * lambda/nbOfTrainingExamples
        val grad = error.T.dot(features)  * alpha/nbOfTrainingExamples
        grad(->, 1->) = grad(->, 1->) + regu
        val updatedThetas =  thetas - grad
        println(s"Cost: ${computeCost(features, labels, updatedThetas)}")
        updatedThetas
    })
    updatedTheta

  }


  def computeThetasForEachClass(features: INDArray, labels: INDArray, alpha: Float, iters: Int): INDArray = {
    val thetas = (0 until 10).foldLeft(Nd4j.zeros(10, features.columns())){
      case (allThetas, index) =>
        val labelsDuplicate =  labels.dup()
        val `class` = if(index==0) 10 else index
        //TODO Waiting when Conditions.notEquals will be fixed
        BooleanIndexing.applyWhere(labelsDuplicate, Conditions.equals(`class`), 100)
        BooleanIndexing.applyWhere(labelsDuplicate, Conditions.lessThan(100), 0.0)
        BooleanIndexing.applyWhere(labelsDuplicate, Conditions.equals(100), 1.0)
        val currentThetas = computeGradient(features, labelsDuplicate, alpha, iters)
        allThetas(index, ->) = currentThetas
    }
    thetas
  }

}


