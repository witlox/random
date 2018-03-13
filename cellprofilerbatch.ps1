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

# inputFile: path to input, use double quotes
# samples: total set of samples to divide over the amount of processes
# jobCount: number of jobs to spawn (0 = as many as we have CPU's)
# application: application to run
# start: offset of the samples to start at
param([String]$inputFile="G:\Batch_data.h5", [Int32]$samples=92160, [Int32]$jobCount=0, [String]$application="C:\Program Files\CellProfiler\CellProfiler.exe", [Int32]$start=1)

Write-Debug "clearing old cache"
Get-Job -Name cp-* | Remove-Job -Force

Write-Output "starting cellprofiler batch run"
if ($jobCount -eq 0) {
    Write-Output "no jobCount specified, detecting number of cores"
    $processors = Get-WmiObject -ComputerName localhost win32_processor
    if (@($processors)[0].NumberOfCores) {
        $cores = @($processors).count * @($processors)[0].NumberOfCores
    } else {
        $cores = @($processors).count
    }
    $jobCount = $cores
}
Write-Debug "setting number of jobs to $jobCount"
$chunk = [math]::Ceiling(($samples-($start-1))/$jobCount)
Write-Debug "chunk size determined as $chunk"

for ($index = $start - 1; $index -lt $samples; $index += $chunk) {
    $start = $index + 1
    $end = [math]::min(($index + $chunk), $samples)
    Write-Output "Starting $application -p $inputFile -c -r -f $start -l $end"
    $block = {
        param ([String]$application, [String[]]$arguments)
        & $application $arguments
    }
    Start-Job -Name "cp-$start-$end" -ScriptBlock $block -ArgumentList @($application, @("-p", $inputFile, "-c", "-r", "-f", $start, "-l", $end))
}

try {
    [console]::TreatControlCAsInput = $true
    Do {
        if ([console]::KeyAvailable) {
            $key = [system.console]::readkey($true)
            if (($key.modifiers -band [consolemodifiers]"control") -and ($key.key -eq "C")) {
				Write-Output "Terminating on request..."
				$abort = $true
				break
            }
        }
        Write-Output "currently $(@(Get-Job -Name cp-* | Where {$_.State -eq "Running"}).Count) / $jobCount active"
        Get-Job -Name cp-* | ForEach-Object {
            Write-Debug "[$($_.State)] $($_.Name) $($_.Progress)"
        }
        Start-Sleep 3
    } While (@(Get-Job -Name cp-* | Where {$_.State -eq "Running"}).Count -ne 0)

	Get-Job -Name cp-* | ForEach-Object {
		Write-Output "[$($_.State)] $($_.Name)"
		if ($_.State -eq "Failed") {
			Receive-Job -Job $_ | Set-Content "$($_.Name).err"
			Write-Host "trying to write results for $($_.Name) to $($_.Name).err" -ForegroundColor Red
		} else {
			Receive-Job -Job $_ | Set-Content "$($_.Name).out"
			Write-Host "trying to write results for $($_.Name) to $($_.Name).out" -ForegroundColor Green
		}
	}
} finally {
    Get-Job -Name cp-* | ForEach-Object { return "[$($_.State)] $($_.Name)" } | Set-Content "run-state.out"
    Get-Job -Name cp-* | Remove-Job -Force
    [console]::TreatControlCAsInput = $false
    Write-Output "finished"
}
