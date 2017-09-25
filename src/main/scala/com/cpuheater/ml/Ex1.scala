package com.cpuheater.ml

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._


object Ex1  extends App{


  val numLinesToSkip = 0
  val delimiter = ","
  val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
  recordReader.initialize(new FileSplit(new ClassPathResource("ex1/ex1data1.txt").getFile))


  val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,1,1, true)
  val dataSet: DataSet = iter.next()


  val ex1data1Features = dataSet.getFeatures()
  val ex1data1Labels = dataSet.getLabels()


  val ones = Nd4j.onesLike(ex1data1Features)
  val ex1data1FeaturesWithBias =  Nd4j.concat(1, ones, ex1data1Features)



  val thetas =  Nd4j.create(Array(0d, 0d)).reshape(1, 2)



  def computeCost(features: INDArray, labels: INDArray, theta: INDArray): Float = {
    val r = pow((features.mmul(theta.T)) - labels, 2)
    val r2 = r.sum(0)/(2*r.length)
    r2.getFloat(0)
  }


  def computeGradient(features: INDArray, labels: INDArray, theta: INDArray, alpha: Double, iters: Int): INDArray ={
    val temp = Nd4j.zerosLike(theta)
    val params = theta.length()
    val cost = Nd4j.zeros(iters)

    val updatedTheta = (0 to iters).foldLeft(temp)({
      case (accum, i) =>
        val error = (features.mmul(accum.T)) - labels
        (0 until params).map{
           p =>
             val r1 = error * features.getColumn(p)
             val d = r1.sum(0).mul(alpha/features.rows()).getFloat(0)
             val r2 =  accum.getFloat(0, p) - r1.sum(0).mul(alpha/features.rows()).getFloat(0)
            accum.put(0, p, r2)
        }
        println(s"Cost: ${computeCost(features, labels, accum)}")
        accum
    })
    updatedTheta

  }

  val alpha = 0.01
  val iterations = 1500

  val computedThetas = computeGradient(ex1data1FeaturesWithBias, ex1data1Labels, thetas, alpha, iterations)
  /**
    * y = theta1*x + theta0
    */

  println(s"theta0 = ${computedThetas.getColumn(0)} theta1=${computedThetas.getColumn(1)}")


  println("Linear regression with multiple variables")


  /**
    * Feature normalization - subtract mean, divide by standard deviation
    *
  **/


}
