package com.cpuheater.ml

import com.cpuheater.util.Loader
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4s.Evidences.float
import org.nd4s.Implicits._

import scala.util.Random

object Ex6 extends App{

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)


  def accuracy(pred: INDArray, labels: INDArray): Float = {
    val sum = (0 until pred.rows()).map{
      index =>
        if(pred.getRow(index).getFloat(0) == labels.getRow(index).getFloat(0))
          1
        else
          0
    }.sum
    sum.toFloat / pred.rows()
  }


  def linearSVM(): Unit = {
    val content = Loader.load("ex6/ex6data1.mat")
    val features: INDArray =  Nd4j.create(content("X"))
    val labels:INDArray = Nd4j.create(content("y"))
    BooleanIndexing.applyWhere(labels, Conditions.lessThan(1), -1)
    val svm = new SVM(10, 0.001f)
    svm.fit(features, labels)
    val pred = svm.predict(features)

    println(s"Accuracy: ${accuracy(pred.T, labels)}")
  }

  linearSVM()
}


class SVM(iter: Int, epsilon: Float = 0.001f) {


  val c: Float = 1.0f

  var w = Nd4j.zeros(1)
  var b = 0f

  def fit(features: INDArray, labels: INDArray) : Unit = {
    val Array(n, d) = features.shape()
    val alpha = Nd4j.zeros(n)
    var count = 0
    var continue = true
    while(continue) {
      count = count +1
      val alpha_prev = alpha.dup()
      (0 until n).map{
         j =>
          val i = randi(0, n-1, j)
          val (x_i, x_j, y_i, y_j) = (features(i,->), features(j,->), labels(i).toInt, labels(j).toInt)
          val k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)

          if(k_ij != 0)  {
            val (alpha_prime_j, alpha_prime_i) = (alpha(j), alpha.apply(i))
            val (l, h) = computeMinMax(c, alpha_prime_j, alpha_prime_i, y_j, y_i)

            w = calcW(alpha, labels, features)
            b = calcB(features, labels, w)

            val e_i = e(x_i, y_i, w, b)
            val e_j = e(x_j, y_j, w, b)


            alpha(j) = alpha_prime_j + (y_j * (e_i - e_j))/k_ij
            alpha(j) = Math.max(alpha(j), l)
            alpha(j) = Math.min(alpha(j), h)

            alpha(i) = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha(j))

          }

      }



      val diff =  Nd4j.norm1(alpha - alpha_prev)
      if(diff < epsilon)
        continue = false

      if(count >= iter)
        println(s"Iteration max  ${iter}")
        continue = false
    }

    b = calcB(features, labels, w)
    w = calcW(alpha, labels, features)
  }


  private def h(x: INDArray, w: INDArray, b: Float): INDArray = {
    val result = (w.dot(x.T) + b)
    BooleanIndexing.applyWhere(result, Conditions.lessThan(0), -1)
    BooleanIndexing.applyWhere(result, Conditions.greaterThanOrEqual(0), 1)
    result
  }


  def predict(features: INDArray): INDArray = {
    h(features, w, b)
  }

  private def e(x_k: INDArray, y_k: Int, w: INDArray, b: Float): Float = {
    h(x_k, w, b).getFloat(0).toInt - y_k
  }


  private def calcB(x: INDArray, y: INDArray, w: INDArray): Float ={
    val b_tmp = y - w.dot(x.T)
    Nd4j.mean(b_tmp).getFloat(0)
  }

  private def calcW(alpha: INDArray, y: INDArray, x: INDArray): INDArray = {
    (alpha * y.T).dot(x)
  }



  private def computeMinMax(c: Float, alpha_prime_j: Float, alpha_prime_i: Float, y_j: Float, y_i: Float): (Float, Float) = {
    if(y_i != y_j)
      (Math.max(0, alpha_prime_j - alpha_prime_i), Math.min(c, c - alpha_prime_i + alpha_prime_j))
    else
      (Math.max(0, alpha_prime_i + alpha_prime_j - c), Math.min(c, alpha_prime_i + alpha_prime_j))
  }

  private def kernel(x1: INDArray, x2: INDArray): Float ={
    x1.dot(x2.T).getFloat(0)
  }


  private def randi(a: Int,b: Int,z:Int) : Int = {
    var i = z
    while (i == z)
      i = a + Random.nextInt(a+b)
    i
  }

}
