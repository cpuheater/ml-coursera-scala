package com.cpuheater.ml

import com.cpuheater.util.Loader
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4j.linalg.ops.transforms.Transforms._

import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.indexing.functions.Value
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.conditions.Conditions

object Ex8 extends App {

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)



  def calcGauss(features: INDArray): (INDArray, INDArray) = {
    val mu = features.mean(0)
    val sigma = features.`var`(0)
    (mu, sigma)
  }


  def estGauss(features:INDArray): (INDArray, INDArray) = {
    val mu = features.mean(0)
    val variance = features.`var`(0)
    (mu, variance)
  }


  def estMultivariateGauss(features: INDArray) : (INDArray, INDArray) = {
    val mu = features.mean(0)
    val nbExamples = features.rows()
    val mubroadcast =   mu.broadcast(Array(features.rows(), mu.columns()): _*)
    val sigma2 = ((features-mubroadcast).T.dot(features-mubroadcast))/nbExamples
    (mu, sigma2)
  }






  def calcPDF(features: INDArray, mu: INDArray, sigma: INDArray): INDArray = {

    val det = sigma.getDouble(0,0)*sigma.getDouble(1,1) - sigma.getDouble(0,1)*sigma.getDouble(1,0)

    val n = features.columns()

    if(sigma.isVector())
        Nd4j.diag(sigma)
    val norm =  1.0/(Math.sqrt(det) *  (Math.pow(2*Math.PI, n/2)))

    val density = Nd4j.zeros(features.rows(), 1)
    val inv =  InvertMatrix.invert(sigma, false)
    (0 until features.rows()).map{
      index =>
        val feature = features.getRow(index)
        val term = (feature-mu).dot(inv).dot((feature-mu).T) * (-0.5)
        density(index, 0) =  exp( term) * norm
    }
    density
  }



  def computeF1(predictions: INDArray, labels: INDArray) : Double= {

    val P = if(predictions.sum(0).getDouble(0) > 0) {
      (0 until predictions.rows()).foldLeft(0) {
        case (p, index) =>
          val pred = predictions.getRow(index).getDouble(0)
          val label = labels.getRow(index).getDouble(0)
          if (label == 1.0 && label == pred)
            p + 1
          else
            p

      } / predictions.sum(0).getDouble(0)
    } else {
      0
    }

    val R = if(predictions.sum(0).getDouble(0) > 0) {
      (0 until predictions.rows()).foldLeft(0) {
        case (r, index) =>
          val pred = predictions.getRow(index).getDouble(0)
          val label = labels.getRow(index).getDouble(0)
          if (label == 1.0 && label == pred)
            r + 1
          else
            r

      } / labels.sum(0).getDouble(0)
    } else {
      0
    }

    if(P > 0 && R > 0)
      2*P*R/(P+R)
    else
      0
  }


  def selectThreshold(labelsVal: INDArray, predictions: INDArray) = {
    val steps = 1000


    val epses: INDArray = Nd4j.linspace(predictions.min(0).getDouble(0),
                                        predictions.max(0).getDouble(0), steps)

    (0 until epses.columns()).foldLeft((0.0, 0.0)){
      case ((bestF1, bestEPs), i) =>
        val eps = epses.getColumn(i).getDouble(0)
        val predictionsDuplicate = predictions.dup()
        BooleanIndexing.applyWhere(predictionsDuplicate, Conditions.lessThan(eps), 1.0d)
        BooleanIndexing.applyWhere(predictionsDuplicate, Conditions.lessThan(1.0d), 0.0d)
        val tempF1 = computeF1(predictionsDuplicate, labelsVal)
        if(tempF1 > bestF1) {
          (tempF1, eps)
        }
        else {
          (bestF1, bestEPs)
        }
    }


  }





  def anomalyDetection() = {

    val data = Loader.load("ex8/ex8data1.mat")
    val features =  Nd4j.create(data("X"))

    val featureValidation = Nd4j.create(data("Xval"))
    val labelsValidation = Nd4j.create(data("yval"))

    val probs = Nd4j.zerosLike(features)

    val probsValidation = Nd4j.zeros(featureValidation.rows(), 1)

    val (mu, sigma) = estMultivariateGauss(features)

    probsValidation(->,->) = calcPDF(featureValidation, mu, sigma)


    val (bestF1, bestEps) = selectThreshold(labelsValidation, probsValidation)

    println(s"Best epsilon: ${bestEps}, best F1: ${bestF1}")

  }

  def recommendation(): Unit = {
    val data = Loader.load("ex8/ex8data2.mat")
    val y =  Nd4j.create(data("Y"))
    val r =  Nd4j.create(data("R"))

    val params = Loader.load("ex8/ex8_movieParams.mat")

    val X = params("X")
    val Theta = params("Theta")
    val nu = params("num_users")
    val nm = params("num_movies")
    val nf = params("num_features")



  }


  recommendation()




}
