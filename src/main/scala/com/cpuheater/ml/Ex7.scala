package com.cpuheater.ml

import com.cpuheater.util.Loader
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.SortedSet
import scala.util.Random


object Ex7  extends App{

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)


  def distSquared(p1: INDArray, p2: INDArray): Double = {
    val power = pow(p1-p2, 2)
    val dist = Nd4j.sum(power).getDouble(0)
    dist
  }


  def findClosestCentroids(features: INDArray, centroids: INDArray): INDArray = {
    val centroidIndexes = (0 until features.rows()).map{
      index =>
        val feature =  features.getRow(index)
        val (_, centroidIndex) = (0 until centroids.rows()).foldLeft((Double.MaxValue, 0)){
          case ((min, minCentroidIndex), centroidIndex) =>
            val centroid = centroids.getRow(centroidIndex)
            val dist =  distSquared(feature, centroid)
            if(dist < min)
              (dist, centroidIndex)
            else
              (min, minCentroidIndex)
        }
        centroidIndex
    }.toNDArray
    centroidIndexes
  }

  def computeCentroids(features: INDArray, featuresCenIds: INDArray) : INDArray= {

    val uniqueCentroidIds = SortedSet(featuresCenIds.data().asInt(): _*)

    val featuresByCentroid = uniqueCentroidIds.foldLeft(Seq.empty[INDArray]){
      case (accum, cenId) =>
        val featureByCentroid = (0 until features.rows()).foldLeft(Seq.empty[INDArray]){
          case (featureByCentroid, index) =>
            val currentCenId = featuresCenIds.getColumn(index).getInt(0)
            val feature = features.getRow(index)
            if(cenId == currentCenId) {
              featureByCentroid :+ feature
            }
            else
              featureByCentroid
        }
        accum :+ Nd4j.create(featureByCentroid, Array(featureByCentroid.size, featureByCentroid(0).columns()))
    }

    featuresByCentroid.zipWithIndex.foldLeft(Nd4j.create(uniqueCentroidIds.size, features.getRow(0).length())) {
      case (accum, (features, index)) =>
        val ndMean = features.mean(0)
        accum.putRow(index, ndMean)
        accum
    }
  }

  def runKMeans(features: INDArray, centroids: INDArray, k: Int, iter: Int) : INDArray= {
    (0 until iter).foldLeft(centroids){
      case (currentCentroid, _) =>
        val newCenIds = findClosestCentroids(features, currentCentroid)
        val newCentroids = computeCentroids(features, newCenIds)
        newCentroids
    }
  }



  def chooseKRandomCentroids(features: INDArray, k: Int): INDArray = {
    val tmp = Nd4j.zeros(Array(k, features.getRow(0).columns()): _*)
    (0 until tmp.rows()).foreach{
      index =>
        val random = Random.nextInt(features.rows())
        tmp.putRow(index, features.getRow(random))
    }
    tmp

  }

  def featureNormalize(features: INDArray):  (INDArray, INDArray, INDArray) = {
    val mean = Nd4j.mean(features, 0)
    val norm = features.subRowVector(mean)
    val std = Nd4j.std(norm, 0)
    val features_normalize = norm.divRowVector(std)
    (mean, std, features_normalize)
  }

  def getUSV(features: INDArray): (INDArray, INDArray, INDArray) = {
    val covMatrix = features.T.dot(features)/features.rows()

    val m = features.rows
    val n = features.columns


    val s = Nd4j.create(if (m < n) m else n)
    val u = if (m < n) Nd4j.create(n, n) else Nd4j.create(n, n)
    val v = Nd4j.create(n, n)

    Nd4j.getBlasWrapper.lapack.gesvd(covMatrix, s, u, v)


    (u, s, v)
  }


  def kMeansClustering()  {

    val data = Loader.load("ex7/ex7data2.mat")
    val features: INDArray =  Nd4j.create(data("X"))

    val K = 3

    val centroids = Nd4j.create(Array(Array(3.0, 3), Array(6.0,2), Array(8.0,5)))

    val newCentroids = runKMeans(features, centroids , 3, 10)
    println(s"Centroids: ${newCentroids}")


  }

  def imageCompressionKMeans()  {

    val data = Loader.load("ex7/bird_small.mat")
    val features: INDArray =  (Nd4j.create(data("A")) / 255).reshape(-1, 3)


    val K = 16
    val randomCentroids = chooseKRandomCentroids(features,K)
    val newCentroids = runKMeans(features,randomCentroids, K, 10)

    println(s"Image compression centroids : ${newCentroids}")


  }

  def projectData(features: INDArray, u: INDArray, k: Int) : INDArray = {
    val uReduced = u(->, 0 -> k)
    features.dot(uReduced)
  }

  def recoverData(z: INDArray, u: INDArray, k: Int) = {
    val uReduced = u(->, 0 -> k)
    z.dot(uReduced.T)
  }

  def pca(): Unit = {

    val data = Loader.load("ex7/ex7data1.mat")
    val features: INDArray =  (Nd4j.create(data("X")))

    val (means, std, featuresNorm) = featureNormalize(features)


    val (u, s, v) = getUSV(featuresNorm)
    println(s"Matrix U ${u}")
    println(s"Matrix S ${s}")
    println(s"Matrix V ${v}")


    val z = projectData(featuresNorm, u, 1)
    println(s"Projections of first components ${z(0)}")


    val recovered = recoverData(z, u, 1)
    println(s"Recovered data ${recovered}")


  }



  def faces(): Unit = {

    val data = Loader.load("ex7/ex7faces.mat")
    val features: INDArray =  (Nd4j.create(data("X")))

    val (means, std, featuresNorm) = featureNormalize(features)


    val (u, s, v) = getUSV(featuresNorm)


    val z = projectData(featuresNorm, u, 1)
    println(s"Projections of first components ${z(0)}")


    val recovered = recoverData(z, u, 1)
    println(s"Recovered data ${recovered.getRow(0)}")


  }

  //kMeansClustering()

  //imageCompressionKMeans()

  pca()

 // faces()














}



