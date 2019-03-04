val commonSettings = Seq(
  scalaVersion := "2.12.8",
  libraryDependencies ++= Seq(
    "edu.berkeley.cs" %% "chisel3" % "3.1.5",
    "edu.berkeley.cs" %% "chisel-iotesters" % "1.2.+",
    "org.scalatest" %% "scalatest" % "3.0.1"
  ),
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases")
  )
)


val vtaSettings = commonSettings ++ Seq(
  name := "chisel-vta",
  version := "0.1-SNAPSHOT",
  organization := "org.liangfu")

lazy val lib = project settings commonSettings
lazy val vta = project in file(".") settings vtaSettings dependsOn lib
