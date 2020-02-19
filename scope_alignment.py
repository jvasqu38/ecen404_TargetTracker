# needed libraries
# import math and csv libraries
import math
import csv
# import pandas for dataframe 
import pandas as pd


# needed dataframes
# import preloaded bullet data for most common rifle cartridges
df = pd.read_excel('bulletData.xlsx')
# import drop calculations from ballisitic drops spreadsheet
df2 = pd.read_csv('drop.csv')

# needed constants:
# form factor for bullets of diameter .22 - .3
I = 0.920


# most common distances include 100, 200, and 300 yards for rifle scopes
# function to calculate time of flight
# formula for Time of Flight = (2*dInFeet)/(muzzle velocity + downrange velocity)
# calculate time of flights for all of the cartridges in dataframe
def calcTOF(d):

    # converts distance from yards to feet
    dInFeet = d*3.0
    
    # creat list for time of flight (tof)
    tof = []
    
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # if distance is 100 yards then pull from 100 yard velocity column
        if(d == 100):  
            # append to list
            tof.append(2*dInFeet/(df['Muzzle Velocity (f/s)'][i]+df['100 Yd Velocity (f/s)'][i]))
        # if distance is 200 yards then pull from 200 yard velocity column
        elif(d == 200):
            # append to list 
            tof.append(2*dInFeet/(df['Muzzle Velocity (f/s)'][i]+df['200 Yd Velocity (f/s)'][i]))
        # else the distance is 300 yards so pull from 300 yard velocity column
        else:
            # append to list
            tof.append(2*dInFeet/(df['Muzzle Velocity (f/s)'][i]+df['300 Yd Velocity (f/s)'][i]))
    # append time of flight calculation to dataframe 
    df['Time of Flight (secs) @ ' + str(d) + ' yards'] = tof
    

# function to calculate the ballistic coefficient 
# formula for Ballistic Coefficient = (bullet weight in pds)/(form factor * diameter of bullet^2)
def calcBallisticCoefficient(I):
    # create list for ballisitic coefficient values
    bc = []
    
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # weight in grains has to converted to weight in pounds by dividing by 7000
        # append to list
        bc.append((df['Weight (Grains)'][i]/7000)/(I*(df['Diameter (Inches)'][i]**2)))
    # append ballistic coefficient calculation to dataframe
    df['Ballistic Coefficient'] = bc


# function to calcuate the drop of the bullet using the ballisitic drop constants spreadsheet
# formula for the Drop = ratio of terminal and intital velocities * (time of flight^2)
def calcDrop(d):
    # create list for drop values
    drop = []
    
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # ratio of terminal and initial velocities rounded to 2 places
        val = round(df['100 Yd Velocity (f/s)'][i]/df['Muzzle Velocity (f/s)'][i],2)
        # find the drop value corresponding to the ratio from the spreadsheet
        f = df2['f'].where(df2['V/Vo'] == val).dropna().tolist()[0]
        # append to list
        drop.append(f*df['Time of Flight (secs) @ ' + str(d) + ' yards'][i]**2)
    # append drop calculation to dataframe
    df['Distance Dropped (Inches) @ ' + str(d) + ' yards'] = drop

    
# function to calculate maximum ordinate of projectile
# describes maximum distance of round above line of sight
# formula for the maximum ordiante = 48*(time of flight^2)
def calcMaxOrd(d):
    # create list for maximum ordinate
    h = []
    
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # append to list
        h.append(48*df['Time of Flight (secs) @ ' + str(d) + ' yards'][i]**2)   
    # append maximum ordinate calculation to dataframe
    df['Maximum Ordinate (Inches)@ ' + str(d/2) + ' yards'] = h


# function to calculate angle of departure
# formula for AOD = 60*(180*((MaxOrd + Drop) / 48) / π)
def calcAOD(d):
    # create list for AOD
    AOD = []
    
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # store half the calculation in val
        val = df['Maximum Ordinate (Inches)@ ' + str(d/2) + ' yards'][i] + df['Distance Dropped (Inches) @ ' + str(d) + ' yards'][i]
        # calculate the second half
        tanRad = val / (48*d)
        # append to list
        AOD.append((tanRad * 180 / math.pi) * 60.0)
    # append AOD calculation to dataframe    
    df['Angle of Departure (MOA)'] = AOD


# function to calculate drift 
# formula to calculate drift = wind_speed*(time of flight - distance/terminal velocity)*wind_angle
def calcDrift(d, wind_speed):
    # create list for drift
    drift = []
    
    # re-instantiate wind speed after user input
    w = wind_speed
    # wind angle
    angle = 20
    
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # calculates half the calculation
        x = df['Time of Flight (secs) @ ' + str(d) + ' yards'][i]-d/df.values[:, df.columns.str.startswith(str(d))].tolist()[i][0]
        # append to list
        drift.append(w*(x)*math.sin(math.radians(20)))
    # append drift calculation to dataframe  
    df['Drift (Inches) @ ' + str(d) + ' yards'] = drift



# function to calculate elevation for the first session
# formula for elevation = AOD/(minute of angle)
def calcElevationPrimary(scope_MOA):
    # create list for elevation
    e = []
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # append to list
        e.append(round(df['Angle of Departure (MOA)'][i]/scope_MOA))
    # append elevation primary to dataframe
    df['Adjust Elevation Up (Clicks)'] = e


# function to calculate windage for the first session
# formula for elevation = Drift/(minute of angle)
def calcWindagePrimary(d, scope_MOA, wind_direction):
    # create list for windage
    windage = []
    # loop that runs for the lenth of the dataframe (since all columns have same # of values)
    for i in range(len(df)):
        # append to list
        windage.append(round(df['Drift (Inches) @ ' + str(d) + ' yards'][i]/scope_MOA))
    # if the wind direction is left, then adjust windage right
    if wind_direction == 'left':
        # append windage primary to dataframe
        df['Adjust Windage Right (Clicks)'] = windage
    # if the wind direction is right, then adjust windage left
    else:
        # append windage primary to dataframe
        df['Adjust Windage Left (Clicks)'] = windage


    
# for continuing sessions
# function to calculate elevation for continuing sessions
# formula for elevation = y_inches/(minute of angle)
def calcElevationSecondary(y_inches, scope_MOA):
    # create list for elevation for continuing sessions
    e2 = []
    # if distance for y_inches is less than 0, then adjust elevation up
    if y_inches < 0:
        # append to list
        e2.append('Adjust elevation control ' + str(abs(round(y_inches/scope_MOA))) + ' clicks up')
    # if distance for y_inches is greater than 0, then adjust elevation down
    else:
        # append to list
        e2.append('Adjust elevation control ' + str(abs(round(y_inches/scope_MOA))) + ' clicks down')
    # return the number of clicks from first index of list
    return e2[0]

# function to calcuation windage for continuing sessions
# formula for windage = x_inches/(minute of angle)
def calcWindageSecondary(x_inches, scope_MOA):
    # create list for windage for continuing sessions
    w2 = []
    # if distance for x_inches is less than 0, then adjust windage right
    if x_inches < 0:
        # append to list
        w2.append('Adjust windage control ' + str(abs(round(x_inches/scope_MOA))) + ' clicks right')
    # if distance for x_inches is greater than 0, then adjust windage left
    else:
        # append to list
        w2.append('Adjust windage control ' + str(abs(round(x_inches/scope_MOA))) + ' clicks left')
    # return the number of clicks from first index of list
    return w2[0]


def runProg():
    # input for session which determines where to start calculation depending on session
    session = int(input('Please input (1) if new session or (2) if continuing a session: '))
    
    # input for distance to target
    d = int(input('Please enter distance from where you are shooting at (100, 200, or 300 yards): '))
    
    # input for rifle minute-of-angle measurement metric for user's scope
    scope_MOA = float(input('Please input MOA measurement, as a decimal, for your scope (most common is 1/4 MOA or 0.25): '))
    
    # if the sesssion is the first session
    if session == 1:
        # input for crosswind speed in mph
        wind_speed = int(input('Please enter the crosswind speed, as a whole number (typical crosswinds are about 5-10 mph) in mph: '))
    
        # input for crosswind direction, i.e., left or right
        wind_direction = input('Please enter direction of crosswind, that is, left or right: ')
        # call function TOF with distance as a parameter
        calcTOF(d)
        # call function ballisitic coefficient with form factor as a parameter
        calcBallisticCoefficient(I)
        # call function drop with distance as a parameter
        calcDrop(d)
        # call function max ord with distance as a parameter
        calcMaxOrd(d)
        # call function AOD with distance as a parameter
        calcAOD(d)
        # call function drift with distance and wind_speed as parameters
        calcDrift(d, wind_speed)
        # call function elevation primary with scope's MOA as a parameter
        calcElevationPrimary(scope_MOA)
        # call function windage primary with scope's MOA as a parameter
        calcWindagePrimary(d, scope_MOA, wind_direction)
        return df
    # if the session is a continued session
    else:
        # input for distance in inches that shot misses target in x-direction
        x_inches = float(input('Please enter distance shot missed in x-direction (- if left and + if right) in inches: '))
    
        # input for distance in inches that shot misses target in y-direction
        y_inches = float(input('Please enter distance shot missed in y-direction (- if down and + if up) in inches: '))
        
        # store elevation and windage adjustment clicks in respective variables
        e2 = calcElevationSecondary(y_inches, scope_MOA)
        w2 = calcWindageSecondary(x_inches, scope_MOA)
        
        # return number of clicks
        return (e2, w2)
        
# run program
runProg()