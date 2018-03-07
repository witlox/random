#   cellprofilerbatch.ps1 -- Run cellprofiler in parallel batch mode
#
#   Copyright (C) 2018 University of Zurich
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

# inputfile: path to input, use double quotes
# samples: total set of samples to devide over the amount of processes
# numproc: number of processes to spawn (0 = as many as we have cpu's)
# app: application to run
# start: offset of the samples to start at
param([String]$inputfile="G:\Batch_data.h5", [Int32]$samples=92160, [Int32]$numproc=0, [String]$app="C:\Program Files\CellProfiler\CellProfiler.exe", [Int32]$start=1)

function Get-CPUs {
    $processors = get-wmiobject -computername localhost win32_processor
    if (@($processors)[0].NumberOfCores) {
        $cores = @($processors).count * @($processors)[0].NumberOfCores
    } else {
        $cores = @($processors).count
    }
    return $cores
}

workflow CellProfiler-Batch {
    param([String[]]$jobs, [String]$app)
    $jc = $jobs.Length
    Write-Output "got $jc jobs"
    $processes = @()
    foreach -parallel ($job in $jobs) {
         Write-Output "Going to invoke $app with arguments: $job"
         $WORKFLOW:processes += Start-Process -FilePath $app -ArgumentList $job -PassThru
    }
    return $processes
}

Write-Output "preparing cellprofiler batch run"
if ($numproc -eq 0) { 
    $cc = Get-CPUs
    Write-Output "detected $cc cores, going to create $cc processes"
} else {
    $cc = $numproc
    Write-Output "using specified number of processes $cc"
}

$chunk = [math]::Ceiling(($samples-($start-1))/$cc)
Write-Output  "Using chunk size $chunk"

$commands = @()
for ($index=$start-1; $index -lt $samples; $index+=$chunk) {
    $first = $index+1
    $last = [math]::min(($index + $chunk),$samples)
    $commands += "-p $inputfile -c -r -f $first -l $last"
}

$processes = CellProfiler-Batch $commands $app

$running = 0
Do {
    $running = 0
    Write-Output "checking background processes"
    foreach ($process in $processes) {
        if (!$process.HasExited) {
            $running++
        }
    }
    Write-Output "$running processes alive"
    Start-Sleep 10
} While ($running -ne 0)

Write-Output "finished"
