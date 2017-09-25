package com.cpuheater.util

import java.io.File

import com.jmatio.io.MatFileReader
import com.jmatio.types.MLDouble
import scala.collection.JavaConversions._

object Loader {

   def load(fileName: String): Map[String, Array[Array[Double]]] = {
     val classLoader = getClass.getClassLoader
     val file = new File(classLoader.getResource(fileName).getFile)
     val mfr = new MatFileReader(file)
     mfr.getContent.map{ case (key, array) => (key, array.asInstanceOf[MLDouble].getArray)}.toMap
   }

}
