name := "ml-coursera-scala"
organization := "com.cpuheater"
version := "0.0.1"

scalaVersion in ThisBuild := "2.11.8"


resolvers +=
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"


val nd4jVersion = "0.7.2"

libraryDependencies ++= Seq(
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
   "org.nd4j" %% "nd4s" % nd4jVersion,
  "org.datavec"        %  "datavec-api" % nd4jVersion,
  "org.deeplearning4j" % "deeplearning4j-core" % nd4jVersion,
  "com.diffplug.matsim" % "matfilerw" % "3.0.1"
)
