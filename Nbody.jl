workspace()


C = 1.488118299382561e-34 ##Graviational Constant in AU^3 / kg * d^2

##Body structure containing all essential information for N-Body Problem
struct Body
    name::String
    mass::Float64
    x::Float64
    y::Float64
    z::Float64
    u::Float64
    v::Float64
    w::Float64
    time::Float64
end

##finds distance between 2 bodies i,j
function distance(bodyi,bodyj)
    dis_x = (bodyi.x - bodyj.x)^2
    dis_y = (bodyi.y - bodyj.y)^2
    dis_z = (bodyi.z - bodyj.z)^2
    sqrt(dis_x + dis_y + dis_z)
end

function conv4(n,A,B,C,D)
    sum = 0
    for j = 0:n
        for k = 0:j
            for l = 0:k
                sum = sum + (A[n-j+1]*B[j-k+1]*C[k-l+1]*D[l+1])
            end
        end
    end
    return sum
end

function conv5(n,A,B,C,D,E)
    sum = 0
    for j = 0:n
        for k = 0:j
            for l = 0:k
                for m = 0:l
                    sum = sum + (A[n-j+1] * B[j-k+1] * C[k-l+1] * D[l-m+1] * E[m+1])
                end
            end
        end
    end
    return sum
end

#tried to norm the conv4 and conv5 function into 1 function, but made the program harder to understand, implement, and also slower
#=
##UNUSED
function conv(n,arglist) ##first try for convolution algorithm.df
    sum = 0
    #print arglist
    if n == 0:
        return arglist[0][0]*arglist[1][0]
    for j in range(0,n):
        if len(arglist) <= 2:
            sum = sum + arglist[0][n-j]*arglist[1][j]
        else:
            sum = sum + arglist[0][n-j]*conv(j,arglist[1:])
            ##print "sum = + ", arglist[0][n-j], " * ", conv(j,arglist[1:])
    return sum
end
=#

##Using a list of power coefficients, finds the power series sum at t to degree legnth(powcoef)
function powersum(T, powcoef)
    sum = 0
    for n = 1:length(powcoef)
        sum = sum + powcoef[n]*T[n]
    end
    return sum
end

#shortens time to calculate t^n(1 time instead of 6)
function tpowern(t,n)
    tn = []
    for i = 1:n
        push!(tn,t^i)
    end
    return tn
end

##returns C = {x| x = a[n] - b[n] for a,b in A,B(respectively) at position n}
function lsub(a,b)
    c = []
    for n = 1:length(a)
        push!(c,a[n]-b[n])
    end
    return c
end

Sun = Body(   "Sun", 1.988544e30,-.0071391433,-.0027920198, .0002061839, .0000053742,-.0000074109,-.0000000942,0.0)
Mercury = Body("Mercury",3.302e23,-.1478672233,-.4466929775,-.0231393758, .0211742456,-.0071053864,-.0025229251,0.0)
Venus = Body("Venus",4.8685e24,-.7257693602,-.0252958208, .0413780252, .0005189070,-.0203135525,-.0003072687,0.0)
Earth = Body("Earth",  5.97219e24,-.1756637922, .9659912850, .0002020629,-.0172285715,-.0030150712,-.0000000585,0.0)
Mars = Body("Mars",6.4185e23, 1.383221922,-.0238017408,-.0344118302, .0007533013, .0151788877, .0002996589,0.0)
Jupiter = Body("Jupiter",1898.13e24, 3.996321311, 2.932561211,-.1016170979,-.0045583766, .0064398632, .0000753756,0.0)
Saturn = Body("Saturn",5.68319e26, 6.401416890, 6.565250734,-.3689211141,-.0042851662, .0038845799, .0001025155,0.0)
Uranus = Body("Uranus",86.8103e24, 14.42337843,-13.73845030,-.2379221201, .0026838403, .0026650165,-.0000248423,0.0)
Neptune = Body("Neptune",102.41e24, 16.80361764,-24.99544328, .1274772016, .0025845895, .0017689435,-.0000962938,0.0)

Solarsystem = [Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]

NEEDEDTIME = 6210 # from 01-01-2000 to 01-01-2017
#X = N_body(Solarsystem, NEEDEDTIME, Solarsystem)

function N_bodyIntegrator(bodylist, step, N, inSim = 0, numPoints = 0)
    bodydict = Dict()
    for body in bodylist ##creates a dictionary for easier access to each element
        bodydict[body] = Dict('a'=>[body.x],'b'=>[body.y],'c'=>[body.z],'u'=>[body.u],'v'=>[body.v],'w'=>[body.w])
    end

    r = [] ##list for automatic differentiation variables
    nbody = [] ##list for new bodies after step size
    for i = 1:length(bodylist)
        push!(r,[])
        for j = 1:length(bodylist)
            if i!=j
                r_0 = 1/distance(bodylist[i],bodylist[j])
                push!(r[i],[r_0])
            else
                push!(r[i],[0])
            end
        end
    end
    for n = 1:N
        for j = 1:length(bodylist)
            push!(bodydict[bodylist[j]]['a'],(bodydict[bodylist[j]]['u'][n])/n)
            push!(bodydict[bodylist[j]]['b'],(bodydict[bodylist[j]]['v'][n])/n)
            push!(bodydict[bodylist[j]]['c'],(bodydict[bodylist[j]]['w'][n])/n)
            push!(bodydict[bodylist[j]]['u'],0)
            push!(bodydict[bodylist[j]]['v'],0)
            push!(bodydict[bodylist[j]]['w'],0)

            for m = 1:length(bodylist) ##loop adds all other-planetary force values to coefficients
                if m != j
                    bodydict[bodylist[j]]['u'][n+1] = bodydict[bodylist[j]]['u'][n+1] + (C*bodylist[m].mass*conv4(n-1,lsub(bodydict[bodylist[m]]['a'],bodydict[bodylist[j]]['a']),r[j][m],r[j][m],r[j][m]))/n
                    bodydict[bodylist[j]]['v'][n+1] = bodydict[bodylist[j]]['v'][n+1] + (C*bodylist[m].mass*conv4(n-1,lsub(bodydict[bodylist[m]]['b'],bodydict[bodylist[j]]['b']),r[j][m],r[j][m],r[j][m]))/n
                    bodydict[bodylist[j]]['w'][n+1] = bodydict[bodylist[j]]['w'][n+1] + (C*bodylist[m].mass*conv4(n-1,lsub(bodydict[bodylist[m]]['c'],bodydict[bodylist[j]]['c']),r[j][m],r[j][m],r[j][m]))/n

                    rau = conv5(n-1,lsub(bodydict[bodylist[m]]['a'],bodydict[bodylist[j]]['a']),lsub(bodydict[bodylist[m]]['u'],bodydict[bodylist[j]]['u']),r[j][m],r[j][m],r[j][m])
                    rbv = conv5(n-1,lsub(bodydict[bodylist[m]]['b'],bodydict[bodylist[j]]['b']),lsub(bodydict[bodylist[m]]['v'],bodydict[bodylist[j]]['v']),r[j][m],r[j][m],r[j][m])
                    rcw = conv5(n-1,lsub(bodydict[bodylist[m]]['c'],bodydict[bodylist[j]]['c']),lsub(bodydict[bodylist[m]]['w'],bodydict[bodylist[j]]['w']),r[j][m],r[j][m],r[j][m])
                    push!(r[j][m],0)
                    r[j][m][n+1] = -1*(1/(n+1))*(rau + rbv + rcw)
                end
            end
        end
    end
    if inSim == false
        for k = 1:length(bodylist)
            newx = powersum(step,bodydict[bodylist[k]]['a'])
            newy = powersum(step,bodydict[bodylist[k]]['b'])
            newz = powersum(step,bodydict[bodylist[k]]['c'])
            newu = powersum(step,bodydict[bodylist[k]]['u'])
            newv = powersum(step,bodydict[bodylist[k]]['v'])
            neww = powersum(step,bodydict[bodylist[k]]['w'])
            push!(nbody,Body(bodylist[k].name,bodylist[k].mass,newx,newy,newz,newu,newv,neww,bodylist[1].time + step))
        end

        ##println(nbody[2].x)
        nbodydict = Dict(zip(bodylist,nbody))
        return nbodydict

    else
        nbodydict= Dict()

        T = linspace(bodylist[1].time,step,numPoints)
        for k = 1:length(bodylist)
            nbodydict[bodylist[k]] = [[],[],[],[],[],[]]
            for t = 2:length(T)
                tn = tpowern(T[t],length(bodydict[bodylist[k]]['a']))
                push!(nbodydict[bodylist[k]][1], powersum(tn,bodydict[bodylist[k]]['a']))
                push!(nbodydict[bodylist[k]][2], powersum(tn,bodydict[bodylist[k]]['b']))
                push!(nbodydict[bodylist[k]][3], powersum(tn,bodydict[bodylist[k]]['c']))
                push!(nbodydict[bodylist[k]][4], powersum(tn,bodydict[bodylist[k]]['u']))
                push!(nbodydict[bodylist[k]][5], powersum(tn,bodydict[bodylist[k]]['v']))
                push!(nbodydict[bodylist[k]][6], powersum(tn,bodydict[bodylist[k]]['w']))
            end
        end
        return nbodydict
    end
end

function N_BodySimulator(bodylist,timepass,K,N,numPoints)
    step = timepass/K
    simdict = Dict()
    currenttime = bodylist[1].time
    stepbodydict = Dict()
    stepbodylist = []
    nbodydict = Dict()

    for k = 1:length(bodylist)
        push!(stepbodylist, bodylist[k])
        stepbodydict[bodylist[k]] = bodylist[k]
        simdict[bodylist[k]] = [[bodylist[k].x],[bodylist[k].y],[bodylist[k].z],[bodylist[k].u],[bodylist[k].v],[bodylist[k].w]]
    end

    for m = 1:K
        nbodycurve = N_bodyIntegrator(stepbodylist,step, N, true, numPoints)
        println(m)
        empty!(stepbodylist)
        for body in bodylist
            simdict[body] = [
            [simdict[body][1]; nbodycurve[stepbodydict[body]][1]],
            [simdict[body][2]; nbodycurve[stepbodydict[body]][2]],
            [simdict[body][3]; nbodycurve[stepbodydict[body]][3]],
            [simdict[body][4]; nbodycurve[stepbodydict[body]][4]],
            [simdict[body][5]; nbodycurve[stepbodydict[body]][5]],
            [simdict[body][6]; nbodycurve[stepbodydict[body]][6]]]
            ##println(nbodycurve[stepbodydict[body]][1])

            stepbodydict[body] = Body(body.name,body.mass,simdict[body][1][end],simdict[body][2][end],simdict[body][3][end],simdict[body][4][end],simdict[body][5][end],simdict[body][6][end],currenttime + step)
            push!(stepbodylist,stepbodydict[body])
        end
    end


##Writes all found data to a DAT file
    open("nbodylist.dat", "w") do listbodies
        for body in bodylist
            name = body.name
            write(listbodies, "$name\n")
        end
    end

##TO DO: Add directories for each planet for orginizational purposes

    for body in bodylist
        filename = string(body.name,".dat")
        open(filename, "w") do data
            for i = 1:length(simdict[body][1])
                datax = simdict[body][1][i]
                datay = simdict[body][2][i]
                dataz = simdict[body][3][i]
                datau = simdict[body][4][i]
                datav = simdict[body][5][i]
                dataw = simdict[body][6][i]
                write(data,"$datax $datay $dataz $datau $datav $dataw\n")
            end
        end
    end


    open("nbodycoord.dat", "w") do reset
        write(reset, "N-Body Data:\n")
    end
    open("nbodycoord.dat", "a") do data
        for body in bodylist
            BN = body.name
            write(data, "$BN, $K, $numPoints\n")
            for i = 1:length(simdict[body][1])
                datax = simdict[body][1][i]
                datay = simdict[body][2][i]
                dataz = simdict[body][3][i]
                datau = simdict[body][4][i]
                datav = simdict[body][5][i]
                dataw = simdict[body][6][i]
                write(data,"X: $datax, Y: $datay, Z: $dataz\n")
                write(data,"VX: $datau, VY: $datav, VZ: $dataw\n\n")

            end
        end
    end
    println("closed")
    ##println(stepbodydict[Mercury])
end

N_BodySimulator(Solarsystem, 365,365, 15, 2)

##NOTES
#1 day seems to be the best step size        1 timepass:K
#       larger surpasses radius of convergence

#PROBLEMS
#Any numPoints > 2 results in a growth from 0 to correct state after first iteration of the integrator
#after the first iteration, all further iterations are the same
