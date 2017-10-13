package com.cpuheater.util

import java.io.File

import com.jmatio.io.MatFileReader
import com.jmatio.types._

import scala.collection.JavaConversions._




object Loader {
  /**
    * Helper function to load matlab files
     * @param fileName
    * @return
    */
   def load(fileName: String): Map[String, Array[Array[Double]]] = {
     val classLoader = getClass.getClassLoader
     val file = new File(classLoader.getResource(fileName).getFile)
     val mfr = new MatFileReader(file)
     mfr.getContent.map{
       case (key, array: MLDouble) => (key, array.asInstanceOf[MLDouble].getArray)
       case (key, array: MLUInt8) => (key, array.asInstanceOf[MLUInt8].getArray.map(_.map(_.toDouble)))
     }.toMap
   }

}
