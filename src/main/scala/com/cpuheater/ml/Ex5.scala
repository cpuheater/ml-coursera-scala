package com.cpuheater.ml

import com.cpuheater.util.Loader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.pow
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

object Ex5 extends App {


  val numLinesToSkip = 0
  val delimiter = ","


  def computeCost(features: INDArray, labels: INDArray, thetas: INDArray, lambda: Double = 0.0): Double = {
    val nbTrainingExamples = features.rows()
    val r = pow(hypothesis(features, thetas) - labels, 2)
    val cost = (r.sum(0)* 1.0/(2*nbTrainingExamples)).getDouble(0)
    val regularization =  lambda / (2*nbTrainingExamples) *  thetas(1->).dot(thetas(1->).T).getDouble(0)
    cost + regularization
  }


  def computeGradient(features: INDArray, labels: INDArray, thetas: INDArray, alpha: Double, iters: Int, lambda: Double = 0.0): INDArray ={
    val nbOfTrainingExamples = features.rows
    val updatedThetas = (0 to iters).foldLeft(thetas)({
      case (accum, i) =>
        val error = hypothesis(features, accum) - labels

        val grad =  features.T.dot(error) * alpha/nbOfTrainingExamples
        val regularization = accum*lambda/nbOfTrainingExamples
        regularization(0) = 0
        val ala = grad + regularization
        val updatedThetas = accum - grad + regularization
        println(s"Cost: ${computeCost(features, labels, updatedThetas)}")
        updatedThetas
    })
    updatedThetas

  }




  /**
    * Feature normalization - subtract mean, divide by standard deviation
    *
    **/

  def normalize(features: INDArray): INDArray = {
    val mean = features.mean(0)
    val std = features.std(0)
    val meanBroadcasted =   mean.broadcast(Array(features.rows(), mean.columns()): _*)
    features(->, 1->) =  features(->, 1->) - meanBroadcasted(->, 1->)
    val stdBroadcasted =   mean.broadcast(Array(features.rows(), std.columns()): _*)
    features(->, 1->) = features(->, 1->) / stdBroadcasted(->, 1->)
    features
  }





  /**
    * y = theta1*x + theta0
    */


  def hypothesis(features: INDArray, thetas: INDArray) ={
    features.mmul(thetas)
  }


  def regularizedLinearRegression(): Unit = {

    val alpha = 0.0001
    val iterations = 20000
    val lambda = 0.1


    val content = Loader.load("ex5/ex5data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y"))


    val featuresValidation: INDArray =  Nd4j.create(content("Xval"))
    val labelsValidation:INDArray = Nd4j.create(content("yval"))

    val featuresTest: INDArray =  Nd4j.create(content("Xval"))
    val labelsTest:INDArray = Nd4j.create(content("yval"))




    val featuresWithBias = Nd4j.concat(1, Nd4j.ones(features.rows(), 1), features)

    val thetas = Nd4j.ones(Array(2, 1): _*)

    val cost = computeCost(featuresWithBias, labels,  thetas, 1.0)

    println(s"Regularized cost ${cost}")

    val computedThetas = computeGradient(featuresWithBias, labels, thetas, alpha, iterations, lambda)


    println(s"Computed thetas ${computedThetas}")

  }


  def createPolynomialFeatures(features: INDArray, degree: Int) : INDArray= {
    val featuresDuplicate = features.dup()

    (0 until degree).foldLeft(featuresDuplicate){
      case (accum, index) =>
        val power = index +2
        val newFeaturesDuplicate = Nd4j.concat(1, accum, pow(featuresDuplicate(->, 1), power))
        newFeaturesDuplicate
    }
  }



  def polynomialLinearRegression(): Unit = {

    val alpha = 0.01
    val iterations = 20000
    val lambda = 0.0001




    val content = Loader.load("ex5/ex5data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y"))


    val featuresValidation: INDArray =  Nd4j.create(content("Xval"))
    val labelsValidation:INDArray = Nd4j.create(content("yval"))

    val featuresTest: INDArray =  Nd4j.create(content("Xval"))
    val labelsTest:INDArray = Nd4j.create(content("yval"))




    val featuresWithBias = Nd4j.concat(1, Nd4j.ones(features.rows(), 1), features)


    val newFeatures = createPolynomialFeatures(featuresWithBias, 5)

    val newFeaturesNorm = normalize(newFeatures)

    val thetas = Nd4j.ones(Array(newFeaturesNorm.columns(), 1): _*)

    val computedThetas = computeGradient(newFeaturesNorm, labels, thetas, alpha, iterations, lambda)


    println(s"Computed thetas ${computedThetas}")

  }


  regularizedLinearRegression()


  polynomialLinearRegression()


}
