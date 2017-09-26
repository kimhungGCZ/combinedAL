# ----------------------------------------------------------------------
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""
This file contains plotting tools for NAB data and results. Run this script to
generate example plots.
"""

from nab.plot import PlotNAB
import os


if __name__ == "__main__":
    baseDataDir = "../NAB/results/numenta/artificialWithAnomaly/"
    dataDirTree = os.walk(baseDataDir)
    dirNames = []
    fullFileNames = []
    dataFiles = ()
    dataNames = ()

    for i, dirDescr in enumerate(dataDirTree):
        for fileName in dirDescr[2]:
            fullFileNames.append(dirDescr[0] + "/" + fileName)

    for fileNumber, fullFileName in enumerate(fullFileNames, start=1):
        print("-----------------------------------------")
        fullFileName = fullFileName.split('//')[1].replace('numenta_','')
        print("[ " + str(fileNumber) + " ] " + fullFileName)
        dataFiles = dataFiles + ('artificialWithAnomaly/'+fullFileName,)
        dataNames = dataNames + ('artificialWithAnomaly/'+fullFileName,)

  # To use this script modify one of the code samples below.

  # Sample 1: shows how to plot a set of raw data files with their labels.
  # You can optionally show the windows or probationary period.

    dataFiles = (
      ["realKnownCause/new_data_moisture.csv"])
    dataNames = (
      ["Moisture Sensor Data"])
  #
  # assert len(dataFiles) == len(dataNames)
  #
  # for i in xrange(len(dataFiles)):
  #   dataPlotter = PlotNAB(
  #       dataFile=dataFiles[i],
  #       dataName=dataNames[i],
  #       offline=True,
  #   )
  #   dataPlotter.plot(
  #       withLabels=True,
  #       withWindows=True,
  #       withProbation=True)


  # Sample 2: to plot the results of running one or more detectors uncomment
  # the following and update the list of dataFiles, dataNames, and detectors.
  # Note that you must have run every detector on each data file. You can
  # optionally show the point labels, windows or probationary period. You can
  # also use one of the non-standard profiles.


    detectors=["relativeEntropy"]
    assert len(dataFiles) == len(dataNames)

    allResultsFiles = []
    for f in dataFiles:
        resultFiles = []
        for d in detectors:
            filename = d + "/"+f.replace("/","/"+d+"_")
            resultFiles.append(filename)
        allResultsFiles.append(resultFiles)

    for i in range(len(dataFiles)):
        dataPlotter = PlotNAB(
            dataFile=dataFiles[i],
            dataName=dataNames[i],
            offline=False)
        dataPlotter.plotMultipleDetectors(
            allResultsFiles[i],
            detectors=detectors,
            scoreProfile="standard",
            withLabels=True,
            withWindows=True,
            withProbation=True)
