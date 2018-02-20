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
param([String]$inputfile="G:\Batch_data.h5", [Int32]$samples=92160, [Int32]$numproc=0, [String]$app="C:\Program Files\CellProfiler\CellProfiler.exe")

function Invoke-Process {
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$FilePath,[Parameter()][ValidateNotNullOrEmpty()][string]$ArgumentList)

     Write-Output "Going to invoke $FilePath with arguments: $ArgumentList"

    $ErrorActionPreference = 'Stop'

    try {
        $stdOutTempFile = "$env:TEMP\$([guid]::NewGuid())"
        $stdErrTempFile = "$env:TEMP\$([guid]::NewGuid())"

        $startProcessParams = @{
            FilePath = $FilePath
            ArgumentList = $ArgumentList
            RedirectStandardError = $stdErrTempFile
            RedirectStandardOutput = $stdOutTempFile
            Wait = $true;
            PassThru = $true;
            NoNewWindow = $true;
        }
        if ($PSCmdlet.ShouldProcess("Process [$($FilePath)]", "Run with args: [$($ArgumentList)]")) {
            $cmd = Start-Process @startProcessParams
            $cmdOutput = Get-Content -Path $stdOutTempFile -Raw
            $cmdError = Get-Content -Path $stdErrTempFile -Raw
            if ($cmd.ExitCode -ne 0) {
                if ($cmdError) {
                    throw $cmdError.Trim()
                }
                if ($cmdOutput) {
                    throw $cmdOutput.Trim()
                }
            } else {
                if ([string]::IsNullOrEmpty($cmdOutput) -eq $false) {
                    Write-Output -InputObject $cmdOutput
               }
            }
        }
    } catch {
        $PSCmdlet.ThrowTerminatingError($_)
    } finally {
        Remove-Item -Path $stdOutTempFile, $stdErrTempFile -Force -ErrorAction Ignore
    }
}

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
    param([String]$runner, [String[]]$jobs)
    foreach -parallel ($job in $jobs) {
        Invoke-Process $runner $job
    }
}

Write-Output "preparing cellprofiler batch run"
if ($numproc -eq 0) { 
    $cc = Get-CPUs
    Write-Output "detected $cc cores, going to create $cc processes"
} else {
    $cc = $numproc
    Write-Output "using specified number of processes $cc"
}

$chunk = [math]::Ceiling($samples/$cc)
Write-Output  "Using chunk size $chunk"

$commands = @()
for ($index=0; $index -lt $samples; $index+=$chunk) {
    $first = $index+1
    $last = [math]::min(($index + $chunk),$samples)
    $commands += "-p $inputfile -c -r -f $first -l $last"
}
CellProfiler-Batch $app $commands

Write-Output "finished"
