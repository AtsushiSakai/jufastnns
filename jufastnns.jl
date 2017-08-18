#
# Fast Nearest Neighbor Search on python using kd-tree
#
# author: Atsushi Sakai
#

module jufastnns

using NearestNeighbors
using PyCall

@pyimport matplotlib.pyplot as plt

function printmat(A)
    for i in 1:length(A[1,:])
        println(A[:,i])
    end
end


function test_3d()
    data3d = rand(3, 5000)
    # print(data3d)

    input3d = rand((3, 3))
    # print(input3d)

    kdtree = KDTree(data3d)

    idxs, dists = knn(kdtree, input3d, 1)
    println(idxs)
    println(dists)

end


function test()
    data2d = rand((2, 5000))
    # printmat(data2d)

    input2d = rand((2, 3))
    # printmat(input2d)

    kdtree = KDTree(data2d)

    idxs, dists = knn(kdtree, input2d, 1)
    println(idxs)
    println(dists)

    plt.plot(data2d[1, :], data2d[2, :], ".r")
    plt.plot(input2d[1,:], input2d[2, :], "xk")
    plt.plot([data2d[1, i] for i in idxs], [data2d[2, i] for i in idxs], "xb")
    plt.show()
end


function main()
    println(PROGRAM_FILE," start!!")
    test()
    test_3d()
    println(PROGRAM_FILE," Done!!")
end


if contains(@__FILE__, PROGRAM_FILE)
    @time main()
end

end


