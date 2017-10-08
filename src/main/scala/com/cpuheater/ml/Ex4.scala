package com.cpuheater.ml


import com.cpuheater.util.Loader
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.util.Random


object Ex4  extends App{

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)


  /**
    * Hyperparameters
     */
  val inputSize = 400
  val hiddenSize = 25
  val outputSize = 10
  val learningRate = 1


  val oneHotMap = Nd4j.eye(10)



  def forwardPropagate(feature: INDArray, theta1: INDArray, theta2: INDArray) : (INDArray, INDArray, INDArray, INDArray)= {
    val z2 = feature.mmul(theta1.T)
    val ones = Nd4j.ones(1)
    val a2 = sigmoid(z2)
    val z3 = Nd4j.concat(1, ones, a2).mmul(theta2.T)
    val a3 = sigmoid(z3)
    (z2, a2, z3, a3)
  }




  def computeCost(features: INDArray, labels: INDArray, theta1: INDArray, theta2: INDArray, lambda:Double = 0.0) : Double=  {
    val nbDataExamples = features.rows()
    val totalCost = (0 until nbDataExamples).foldLeft(0.0d){
      case (accu, rowIndex) =>
        val feature =  features.getRow(rowIndex)
        val (z2, a2, z3, a3) = forwardPropagate(feature, theta1, theta2)
        val labelOneHot = oneHotMap.getRow(labels.getRow(rowIndex).getDouble(0).toInt -1)
        val term1 = -labelOneHot.mmul(log(a3.T)).getDouble(0)
        val term2 = -labelOneHot.rsub(1).mmul(log(a3.rsub(1).T)).getDouble(0)
        val cost = term1 + term2
        val regular = Nd4j.sum(theta1*theta1).getDouble(0) + Nd4j.sum(theta2* theta2).getDouble(0)
        val regular_norm =  regular * lambda/(2 * nbDataExamples)
        accu + cost + regular_norm

    }
    totalCost/nbDataExamples
  }



  def sigmoidDerivative(x: INDArray): INDArray = {
    val result =  x*(x.rsub(1))
    result
  }


  def randomThetas() = {
    val epsilonInit = 0.12

    val theta1 = Nd4j.rand(hiddenSize, inputSize+1)  * 2 * epsilonInit - epsilonInit
    val theta2 = Nd4j.rand(outputSize, hiddenSize +1)  * 2 * epsilonInit - epsilonInit

    (theta1, theta2)
  }



  def backpropagation(features: INDArray, labels: INDArray, theta1: INDArray, theta2: INDArray, lambda: Double = 0.0) = {
    val nbDataExamples = features.rows()
    val (delta1, delta2)  = (0 until nbDataExamples).toList.foldLeft((Nd4j.zeros(hiddenSize, inputSize+1), Nd4j.zeros(outputSize, hiddenSize+1))){
      case ((totalDelta1, totalDelta2), rowId) =>
        val feature = features.getRow(rowId)
        val (z2, a2, z3, a3) = forwardPropagate(feature, theta1, theta2)
        val labelOneHot = oneHotMap.getRow(labels.getRow(rowId).getDouble(0).toInt -1)
        val error3 = a3 - labelOneHot
        val error2 =  theta2.T(1->, ->).mmul(error3.T) * sigmoidDerivative(a2)
        val newTotalDelta1 = totalDelta1 + error2.mmul(feature)
        val a2WithOnes = Nd4j.concat(1, Nd4j.ones(1), a2).reshape(1, hiddenSize+1)
        val nnewTotalDelta2 = totalDelta2 + error3.T.mmul(a2WithOnes)
        (newTotalDelta1, nnewTotalDelta2)

    }

    val delta1Norm = delta1/nbDataExamples.toDouble
    val delta2Norm = delta2/nbDataExamples.toDouble

    delta1Norm(->, 1->) = delta1Norm(->, 1->) + theta1(->, 1->)*lambda/nbDataExamples
    delta2Norm(->, 1->) = delta2Norm(->, 1->) + theta2(->, 1->)*lambda/nbDataExamples


    (delta1Norm, delta2Norm)


  }


  def gradientChecking(features: INDArray, labels: INDArray, theta1: INDArray, theta2: INDArray, delta1: INDArray, delta2: INDArray) = {
    val eps = 0.0001

    val nbOfElements = theta1.length() + theta2.length()

    (0 until 5).foreach{
      _ =>
        val randomElement = Random.nextInt(nbOfElements)
        val epsilonVector = Nd4j.create(nbOfElements, 1)
        epsilonVector(randomElement) = eps
        val theta1Epsilon = epsilonVector(0 until (inputSize+1)*hiddenSize).reshape(hiddenSize, inputSize+1)
        val theta2Epsilon = epsilonVector((inputSize +1)*hiddenSize until (inputSize+1)*hiddenSize + outputSize*(hiddenSize+1)).reshape(outputSize, hiddenSize +1)
        val theta1WithEpsilon = theta1 + theta1Epsilon
        val theta2WithEpsilon = theta2 + theta2Epsilon

        val costHigh = computeCost(features, labels, theta1WithEpsilon, theta2WithEpsilon)
        val costLow = computeCost(features, labels, theta1 - theta1Epsilon, theta2 - theta2Epsilon)
        val nG = (costHigh - costLow) / (2 * eps)
        println(s"Element: ${randomElement} calculated numerical gradient: ${nG}, backProp gradient = ${Nd4j.toFlattened(delta1, delta2)(randomElement)}")
    }

  }


  def train(features: INDArray, labels: INDArray, theta1: INDArray, theta2: INDArray, iter: Int, lr: Double = 1, lambda: Double = 0.0) = {

    (0 until iter).foldLeft((theta1, theta2)){
      case ((theta1, theta2), index) =>
        val cost = computeCost(features, labels, theta1, theta2)
        println(s"Cost ${cost}")
        val (delta1, delta2) = backpropagation( features, labels, theta1, theta2, lambda)
        val updatedTheta1 = theta1 - delta1
        val updatedTheta2 = theta2 - delta2
        (updatedTheta1, updatedTheta2)
    }
  }



  def predict(feature: INDArray, theta1: INDArray, theta2: INDArray): Int = {
    val (_, _, _, pred) = forwardPropagate(feature, theta1, theta2)
    val value: Int = Nd4j.argMax(pred).getInt(0) +1
    value
  }



  def computeAccuracy(features: INDArray, labels: INDArray, theta1: INDArray, theta2: INDArray): Double = {
    val nbDataExamples = features.rows()
    val correct = (0 until nbDataExamples).foldLeft(0){
      case (accum, index) =>
        val pred = predict(features.getRow(index), theta1, theta2)
        if(pred == labels(index).toInt)
          accum +1
        else
          accum
    }
    correct.toDouble/nbDataExamples
  }




  def feedforwardNeuralNetwork(): Unit = {

    val content = Loader.load("ex4/ex4data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y"))

    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)

    val data = Loader.load("ex4/ex4weights.mat")

    val (theta1, theta2) = (Nd4j.create(data("Theta1")), Nd4j.create(data("Theta2")))


    val cost  = computeCost(featuresWithBias, labels, theta1, theta2)

    println(s"Cost: ${cost} %")

    val regulCost  = computeCost(featuresWithBias, labels, theta1, theta2, 1)

    println(s"Regularized cost: ${regulCost} %")


  }


  def backpropagationNeuralNetwork(): Unit = {

    val content = Loader.load("ex4/ex4data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y"))

    val ones = Nd4j.ones(features.rows(), 1)
    val featuresWithBias =  Nd4j.concat(1, ones, features)


    val (theta1, theta2) = randomThetas()

    val (delta1, delta2) = backpropagation(featuresWithBias, labels, theta1, theta2)

    gradientChecking(featuresWithBias, labels, theta1, theta2, delta1, delta2)

    val (trainedTheta1, trainedTheta2) = train(featuresWithBias, labels, theta1, theta2, 50, learningRate, 0)



    val accuracy = computeAccuracy(featuresWithBias, labels, trainedTheta1, trainedTheta2)

    println(s"Accuracy: ${accuracy} %")

    val (regTrainedTheta1, regTrainedTheta2) = train(featuresWithBias, labels, theta1, theta2, 50, learningRate, 1)

    val regAccuracy = computeAccuracy(featuresWithBias, labels, trainedTheta1, trainedTheta2)

    println(s"Accuracy with regularization: ${regAccuracy} %")

  }

  feedforwardNeuralNetwork()


  backpropagationNeuralNetwork()




}



