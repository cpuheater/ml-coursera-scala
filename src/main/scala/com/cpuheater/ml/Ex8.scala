package com.cpuheater.ml

import com.cpuheater.util.Loader
import org.datavec.api.records.metadata.RecordMetaData
import org.datavec.api.records.reader.RecordReader
import org.nd4s.Implicits._
import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.conditions.Conditions
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.records.reader.impl.regex.RegexLineRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._


object Ex8 extends App {

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)


  val numLinesToSkip = 0
  val delimiter = ","



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


  def computeCost(features: INDArray, thetas: INDArray, y: INDArray, r: INDArray, numOfUsers: Int, numOfMovies: Int, numOfFeatures: Int, lambda: Double = 0.0) = {
    val r = features.dot(thetas.T)
    val r2 = r.mul(r)
    val cost =Nd4j.sum(pow(r2 - y, 2)) * 0.5

    val reguCost1 = Nd4j.sum(pow(thetas, 2)) * lambda/2.0
    val reguCost2 = Nd4j.sum(pow(features, 2)) * lambda/2.0

    val totalCost = cost + reguCost1 + reguCost2
    totalCost
  }

  def computeGradient(features: INDArray, thetas: INDArray, y: INDArray, r: INDArray, numOfUsers: Int, numOfMovies: Int,
                      numOfFeatures: Int, lambda: Double = 0.0, alpha: Double = 1 ,iters: Int = 1) :( INDArray, INDArray) = {

    val (updatedFeatures, updatedThetas) = (0 until iters).foldLeft((features, thetas)){
      case ((features, thetas), i) =>
        val r1 = features.dot(thetas.T)

        val r2 = r1 * r

        val r3 = r2 - y

        val grad = r3.dot(thetas)

        val thetaGrad = r3.T.dot(features)

        val regGrad = grad + features * lambda

        val regThetaGrad = thetaGrad + thetas * lambda
        val updatedFeatures = features - regGrad * alpha
        val updatedThetas = thetas - regThetaGrad * alpha
        (updatedFeatures, updatedThetas)
    }

    (updatedFeatures, updatedThetas)

  }


  def normalizeRatings(y: INDArray, r: INDArray) = {
    val mean = Nd4j.sum(y, 1) / Nd4j.sum(r, 1)
    val result = y.subColumnVector(mean)
    (result, mean)
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
    val data = Loader.load("ex8/ex8_movies.mat")
    val y =  Nd4j.create(data("Y"))
    val r =  Nd4j.create(data("R"))

    val params = Loader.load("ex8/ex8_movieParams.mat")

    val regex = "(\\d+) (.*)"

    val regexLineRecordReader = new RegexLineRecordReader(regex, 0)
    regexLineRecordReader.initialize(new FileSplit(new ClassPathResource("ex8/movie_ids.txt").getFile))

    val movies = scala.collection.mutable.Map[Int, String]()

    while(regexLineRecordReader.hasNext) {
      val List(id, name) = regexLineRecordReader.next().toList
      movies(id.toInt) = name.toString
    }

    val ratings = Nd4j.zeros(1682,1)
    ratings(0)   = 4
    ratings(97)  = 2
    ratings(6)   = 3
    ratings(11)  = 5
    ratings(53)  = 4
    ratings(63)  = 5
    ratings(65)  = 3
    ratings(68)  = 5
    ratings(182) = 4
    ratings(225) = 5
    ratings(354) = 5

    val yWithNewRatings = Nd4j.concat(1, y, ratings)
    BooleanIndexing.applyWhere(ratings, Conditions.greaterThan(0), 1)
    val rWithNewRatings = Nd4j.concat(1,r, ratings)

    val Array(numOfMovies, numOfUsers) = rWithNewRatings.shape()

    val numOfFeatures = 10

    val features = Nd4j.rand(numOfMovies, numOfFeatures)
    val thetas = Nd4j.rand(numOfUsers, numOfFeatures)

    val (computedFeatures, computedThetas) = computeGradient(features, thetas, yWithNewRatings, rWithNewRatings, numOfUsers, numOfMovies, numOfFeatures, 0, 0.001, 100)


    val predictions = computedFeatures.dot(computedThetas.T)

    val lastPredictions = predictions(->, predictions.columns()-1)

    val Array(indices, newHala) = Nd4j.sortWithIndices(lastPredictions.dup(), 0, false)

    println("Top 10 movies")

    (0 until 10).map{
      i =>
        val index = indices(i).toInt
        println(s" movies ${lastPredictions(index)}, ${movies(index)}")
    }

  }


  recommendation()




}
