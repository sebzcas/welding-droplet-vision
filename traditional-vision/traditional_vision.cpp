#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/geometry.hpp>
#include<iostream>
#include<vector>
#include<cmath>
#include<fstream>
#include <sstream>
#include<array>
#include<algorithm>
#include<new>
#include<stdexcept>
#include<cstdlib>
#include<cstring>
#include<unordered_map>


// All legacy C-style allocations made during one top-level detection call are
// owned by this scope. realloc() is implemented with allocate-copy semantics,
// so a failed allocation cannot overwrite and lose the original pointer.
class AllocationArena
{
public:
    std::vector<void*> blocks;
    std::unordered_map<void*, std::size_t> sizes;

    void* allocate(std::size_t size)
    {
        if (size == 0) size = 1;
        void* ptr = std::malloc(size);
        if (!ptr) throw std::bad_alloc();
        blocks.push_back(ptr);
        sizes[ptr] = size;
        return ptr;
    }

    void* reallocate(void* oldPtr, std::size_t newSize)
    {
        if (!oldPtr) return allocate(newSize);
        if (newSize == 0) newSize = 1;

        const auto it = sizes.find(oldPtr);
        if (it == sizes.end())
            throw std::runtime_error("Attempted to realloc memory not owned by the detection scope.");

        const std::size_t oldSize = it->second;
        void* newPtr = allocate(newSize);
        std::memcpy(newPtr, oldPtr, std::min(oldSize, newSize));
        sizes.erase(oldPtr);
        return newPtr;
    }

    void releaseFrom(std::size_t firstBlock) noexcept
    {
        for (std::size_t i = blocks.size(); i > firstBlock; --i)
        {
            void* ptr = blocks[i - 1];
            std::free(ptr);
            sizes.erase(ptr);
        }
        blocks.resize(firstBlock);
    }
};

static AllocationArena g_allocationArena;

class AllocationScope
{
public:
    AllocationScope() : firstBlock_(g_allocationArena.blocks.size()) {}
    ~AllocationScope() { g_allocationArena.releaseFrom(firstBlock_); }

    AllocationScope(const AllocationScope&) = delete;
    AllocationScope& operator=(const AllocationScope&) = delete;

private:
    std::size_t firstBlock_;
};

#define malloc(size) g_allocationArena.allocate(size)
#define realloc(ptr, size) g_allocationArena.reallocate(ptr, size)
// Individual legacy free() calls are intentionally deferred to the enclosing
// AllocationScope, avoiding invalid frees of shared/intermediate arrays.
#define free(ptr) ((void)0)

using namespace std;
using namespace cv;

int hasWhiteNeighbor(const Mat& srcImage, int x, int y)
{
	int k = 1;
	if(((srcImage.at<Vec3b>(x+k, y)[0]==255)&&(srcImage.at<Vec3b>(x+k, y)[1]==255)&&(srcImage.at<Vec3b>(x+k, y)[2]==255))||
	   ((srcImage.at<Vec3b>(x-k, y)[0]==255)&&(srcImage.at<Vec3b>(x-k, y)[1]==255)&&(srcImage.at<Vec3b>(x-k, y)[2]==255))||
	   ((srcImage.at<Vec3b>(x, y+k)[0]==255)&&(srcImage.at<Vec3b>(x, y+k)[1]==255)&&(srcImage.at<Vec3b>(x, y+k)[2]==255))||
	   ((srcImage.at<Vec3b>(x, y-k)[0]==255)&&(srcImage.at<Vec3b>(x, y-k)[1]==255)&&(srcImage.at<Vec3b>(x, y-k)[2]==255))||
	   ((srcImage.at<Vec3b>(x+k, y+k)[0]==255)&&(srcImage.at<Vec3b>(x+k, y+k)[1]==255)&&(srcImage.at<Vec3b>(x+k, y+k)[2]==255))||
	   ((srcImage.at<Vec3b>(x+k, y-k)[0]==255)&&(srcImage.at<Vec3b>(x+k, y-k)[1]==255)&&(srcImage.at<Vec3b>(x+k, y-k)[2]==255))||
	   ((srcImage.at<Vec3b>(x-k, y-k)[0]==255)&&(srcImage.at<Vec3b>(x-k, y-k)[1]==255)&&(srcImage.at<Vec3b>(x-k, y-k)[2]==255))||
	   ((srcImage.at<Vec3b>(x-k, y+k)[0]==255)&&(srcImage.at<Vec3b>(x-k, y+k)[1]==255)&&(srcImage.at<Vec3b>(x-k, y+k)[2]==255)))
		return 1;
	else
		return 0;
}

int isGreenMarker(const Mat& srcImage, int x, int y)
{
	if(((srcImage.at<Vec3b>(x, y)[0]==0)&&(srcImage.at<Vec3b>(x, y)[1]==255)&&(srcImage.at<Vec3b>(x, y)[2]==0))||
	   ((srcImage.at<Vec3b>(x, y)[0]==144)&&(srcImage.at<Vec3b>(x, y)[1]==238)&&(srcImage.at<Vec3b>(x, y)[2]==144)))
	    return 1;
	else
		return 0;
}

int hasGreenMarkerNearby(const Mat& srcImage, int x, int y, int k)
{
	for(int j=-k;j<=k;j++)
	{
		if(isGreenMarker(srcImage, x+k, y+j)==1)
		{
			return 1;
		}
		else if(isGreenMarker(srcImage, x+j, y+k)==1)
		{
			return 1;
		}
		else if(isGreenMarker(srcImage, x+j, y-k)==1)
		{
			return 1;
		}
		else if(isGreenMarker(srcImage, x-k, y+j)==1)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
}

int* findGreenMarkerNearby(const Mat& srcImage, int x, int y, int k)
{
	static int ptr[3]; int value = 0;
	for(int i=1;i<=k;i++)
	{
		for(int j=-i;j<=i;j++)
		{
			if(isGreenMarker(srcImage, x+i, y+j)==1)
			{
				ptr[0] = 1; ptr[1] = x + i; ptr[2] = y + j; value = 1;
				return ptr;
			}
			else if(isGreenMarker(srcImage, x+j, y+i)==1)
			{
				ptr[0] = 1; ptr[1] = x + j; ptr[2] = y + i; value = 1;
				return ptr;
			}
			else if(isGreenMarker(srcImage, x+j, y-i)==1)
			{
				ptr[0] = 1; ptr[1] = x + j; ptr[2] = y - i; value = 1;
				return ptr;
			}
			else if(isGreenMarker(srcImage, x-i, y+j)==1)
			{
				ptr[0] = 1; ptr[1] = x - i; ptr[2] = y + j; value = 1;
				return ptr;
			}
		}
	}
	if(value==0)
	{
		ptr[0] = 0; ptr[1] = x; ptr[2] = y;
		return ptr;
	}
}

int isBlueOrRedMarker(const Mat& srcImage, int x, int y)
{
	if(((srcImage.at<Vec3b>(x, y)[0]==255)&&(srcImage.at<Vec3b>(x, y)[1]==0)&&(srcImage.at<Vec3b>(x, y)[2]==0))||
	   ((srcImage.at<Vec3b>(x, y)[0]==0)&&(srcImage.at<Vec3b>(x, y)[1]==0)&&(srcImage.at<Vec3b>(x, y)[2]==255)))
	    return 1;
	return 0;
}

void maskBelowRow(Mat& srcImage, int k)
{
	for(int i=k;i<srcImage.rows;i++)
	{
		for(int j=0;j<srcImage.cols;j++)
		{ srcImage.at<Vec3b>(i, j)[0]=0; srcImage.at<Vec3b>(i, j)[1]=0; srcImage.at<Vec3b>(i, j)[2]=0; }
	}
}

int* traceContourStep(Mat& srcImage, int x, int y, int z)
{
	static int ptr[3];
	int x0 = x, y0 = y;

	srcImage.at<Vec3b>(x, y)[0]=0; srcImage.at<Vec3b>(x, y)[1]=0; srcImage.at<Vec3b>(x, y)[2]=255;  // Set the node to red

	z = 9;
	int k = 1; int l = 4;
	for(int i=0;i<z;i++)  // Set each segment to blue
	{
		if((x>=srcImage.rows-l)||(y>=srcImage.cols-l)||(x<=l)||(y<=l))
		{
			// cout<<endl<<"Out of range."<<endl;
			break;
		}
		else
		{
			if((srcImage.at<Vec3b>(x, y+k)[0]==255)&&(srcImage.at<Vec3b>(x, y+k)[1]==255)&&(srcImage.at<Vec3b>(x, y+k)[2]==255)&&(hasWhiteNeighbor(srcImage, x, y+k)==1))
			{
				srcImage.at<Vec3b>(x, y+k)[0] = 255; srcImage.at<Vec3b>(x, y+k)[1] = 0; srcImage.at<Vec3b>(x, y+k)[2] = 0;
				y = y + k;
				//cout<<x<<" "<<y<<endl;
			}
			if((srcImage.at<Vec3b>(x, y-k)[0]==255)&&(srcImage.at<Vec3b>(x, y-k)[1]==255)&&(srcImage.at<Vec3b>(x, y-k)[2]==255)&&(hasWhiteNeighbor(srcImage, x, y-k)==1))
			{
				srcImage.at<Vec3b>(x, y-k)[0] = 255; srcImage.at<Vec3b>(x, y-k)[1] = 0; srcImage.at<Vec3b>(x, y-k)[2] = 0;
				y = y - k;
				//cout<<x<<" "<<y<<endl;
			}
			if((srcImage.at<Vec3b>(x+k, y)[0]==255)&&(srcImage.at<Vec3b>(x+k, y)[1]==255)&&(srcImage.at<Vec3b>(x+k, y)[2]==255)&&(hasWhiteNeighbor(srcImage, x+k, y)==1))
			{
				srcImage.at<Vec3b>(x+k, y)[0] = 255; srcImage.at<Vec3b>(x+k, y)[1] = 0; srcImage.at<Vec3b>(x+k, y)[2] = 0;
				x = x + k;
				//cout<<x<<" "<<y<<endl;
			}
			if((srcImage.at<Vec3b>(x-k, y)[0]==255)&&(srcImage.at<Vec3b>(x-k, y)[1]==255)&&(srcImage.at<Vec3b>(x-k, y)[2]==255)&&(hasWhiteNeighbor(srcImage, x-k, y)==1))
			{
				srcImage.at<Vec3b>(x-k, y)[0] = 255; srcImage.at<Vec3b>(x-k, y)[1] = 0; srcImage.at<Vec3b>(x-k, y)[2] = 0;
				x = x - k;
				//cout<<x<<" "<<y<<endl;
			}
			if((srcImage.at<Vec3b>(x+k, y+k)[0]==255)&&(srcImage.at<Vec3b>(x+k, y+k)[1]==255)&&(srcImage.at<Vec3b>(x+k, y+k)[2]==255)&&(hasWhiteNeighbor(srcImage, x+k, y+k)==1))
			{
				srcImage.at<Vec3b>(x+k, y+k)[0] = 255; srcImage.at<Vec3b>(x+k, y+k)[1] = 0; srcImage.at<Vec3b>(x+k, y+k)[2] = 0;
				x = x + k; y = y + k;
				//cout<<x<<" "<<y<<endl;
			}
			if((srcImage.at<Vec3b>(x+k, y-k)[0]==255)&&(srcImage.at<Vec3b>(x+k, y-k)[1]==255)&&(srcImage.at<Vec3b>(x+k, y-k)[2]==255)&&(hasWhiteNeighbor(srcImage, x+k, y-k)==1))
			{
				srcImage.at<Vec3b>(x+k, y-k)[0] = 255; srcImage.at<Vec3b>(x+k, y-k)[1] = 0; srcImage.at<Vec3b>(x+k, y-k)[2] = 0;
				x = x + k; y = y - k;
				//cout<<x<<" "<<y<<endl;
			}
			if((srcImage.at<Vec3b>(x-k, y+k)[0]==255)&&(srcImage.at<Vec3b>(x-k, y+k)[1]==255)&&(srcImage.at<Vec3b>(x-k, y+k)[2]==255)&&(hasWhiteNeighbor(srcImage, x-k, y+k)==1))
			{
				srcImage.at<Vec3b>(x-k, y+k)[0] = 255; srcImage.at<Vec3b>(x-k, y+k)[1] = 0; srcImage.at<Vec3b>(x-k, y+k)[2] = 0;
				x = x - k; y = y + k;
				//cout<<x<<" "<<y<<endl;
			}
			if((srcImage.at<Vec3b>(x-k, y-k)[0]==255)&&(srcImage.at<Vec3b>(x-k, y-k)[1]==255)&&(srcImage.at<Vec3b>(x-k, y-k)[2]==255)&&(hasWhiteNeighbor(srcImage, x-k, y-k)==1))
			{
				srcImage.at<Vec3b>(x-k, y-k)[0] = 255; srcImage.at<Vec3b>(x-k, y-k)[1] = 0; srcImage.at<Vec3b>(x-k, y-k)[2] = 0;
				x = x - k; y = y - k;
				//cout<<x<<" "<<y<<endl;
			}
		}
	}

	if((x==x0)&&(y==y0))
	{
		//cout<<"Deadlock"<<endl;
		ptr[2] = 1;
	}
	else
		ptr[2] = 0;

	ptr[0] = x; ptr[1] = y;
	srcImage.at<Vec3b>(x, y)[0]=0; srcImage.at<Vec3b>(x, y)[1]=0; srcImage.at<Vec3b>(x, y)[2]=255;
	return ptr;
}

int* traceContourSegment(Mat& srcImage, int x0, int y0, int z)
{
	int *ptr = nullptr;
	int num = 200;
	std::vector<int> stock(3 * num + 4, 0);
	stock[1] = x0; stock[2] = y0; stock[3] = 0;
	//cout<<endl<<"1 "<<"traceContourSegment = "<<stock[1]<<" "<<stock[2]<<" "<<stock[3]<<endl;
 	int i = 1;
	do {
		const std::size_t requiredSize = static_cast<std::size_t>(3 * i + 4);
		if (stock.size() < requiredSize) stock.resize(requiredSize, 0);
		ptr = traceContourStep(srcImage, stock[3*i-2], stock[3*i-1], z);
		stock[3*i+1] = *ptr; stock[3*i+2] = *(ptr+1); stock[3*i+3] = *(ptr+2);
		//line(srcImage, Point(stock[3*i-1], stock[3*i-2]), Point(stock[3*i+2], stock[3*i+1]), Scalar(194, 158, 241), 1);
		//cout<<i+1<<" "<<"traceContourSegment = "<<stock[3*i+1]<<" "<<stock[3*i+2]<<" "<<stock[3*i+3]<<endl;
		if(stock[3*i+3]==1)
		{
            // stock[0] records node number of the segment, whose total length should be (3 * i + 4)
			stock[0] = i + 1;
			srcImage.at<Vec3b>(stock[3*i+1], stock[3*i+2])[0] = 144;
			srcImage.at<Vec3b>(stock[3*i+1], stock[3*i+2])[1] = 238;
			srcImage.at<Vec3b>(stock[3*i+1], stock[3*i+2])[2] = 144;  // Set the end node to green
		}
		i++;
	} while(stock[3*i]==0);
	srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 255; srcImage.at<Vec3b>(x0, y0)[2] = 0;  //起始点置为黄色
	//cout<<stock[0]<<endl;
	int *point; point = (int*)malloc((2*stock[0]-1)*sizeof(int));
	point[0] = stock[0] - 1; //cout<<"  deduplicateTracePoints = "<<point[0]<<endl;
	for(int i=0;i<point[0];i++)
	{
		point[2*i+1] = stock[3*i+1]; point[2*i+2] = stock[3*i+2];
		//cout<<i+1<<" "<<"traceContourSegment = "<<point[2*i+1]<<" "<<point[2*i+2]<<endl;
	} //cout<<endl;

	return point;
}

int** extractContourSegments(Mat& srcImage, int z)
{
	int *ptr = nullptr;
	int **stock;
	int extendContourFromEndpoints = 0; //ptr = traceContourStep(srcImage, 53, 5);
	int x, y;
	int I = srcImage.rows-1, J = srcImage.cols-1;

	stock = (int**)malloc(1*sizeof(int*));
	stock[0] = (int*)malloc(1*sizeof(int));

	for(int j=1;j<J;j++)
	{
		for(int i=1;i<I;i++)
		{
			int x = i, y = j;
			if((srcImage.at<Vec3b>(x, y)[0]==255)&&(srcImage.at<Vec3b>(x, y)[1]==255)&&(srcImage.at<Vec3b>(x, y)[2]==255))
			{
				if(((srcImage.at<Vec3b>(x+1, y)[0]==255)&&(srcImage.at<Vec3b>(x+1, y)[1]==0)&&(srcImage.at<Vec3b>(x+1, y)[2]==0))||
				   ((srcImage.at<Vec3b>(x-1, y)[0]==255)&&(srcImage.at<Vec3b>(x-1, y)[1]==0)&&(srcImage.at<Vec3b>(x-1, y)[2]==0))||
				   ((srcImage.at<Vec3b>(x, y+1)[0]==255)&&(srcImage.at<Vec3b>(x, y+1)[1]==0)&&(srcImage.at<Vec3b>(x, y+1)[2]==0))||
				   ((srcImage.at<Vec3b>(x, y-1)[0]==255)&&(srcImage.at<Vec3b>(x, y-1)[1]==0)&&(srcImage.at<Vec3b>(x, y-1)[2]==0))||
				   ((srcImage.at<Vec3b>(x+1, y+1)[0]==255)&&(srcImage.at<Vec3b>(x+1, y+1)[1]==0)&&(srcImage.at<Vec3b>(x+1, y+1)[2]==0))||
				   ((srcImage.at<Vec3b>(x+1, y-1)[0]==255)&&(srcImage.at<Vec3b>(x+1, y-1)[1]==0)&&(srcImage.at<Vec3b>(x+1, y-1)[2]==0))||
				   ((srcImage.at<Vec3b>(x-1, y-1)[0]==255)&&(srcImage.at<Vec3b>(x-1, y-1)[1]==0)&&(srcImage.at<Vec3b>(x-1, y-1)[2]==0))||
				   ((srcImage.at<Vec3b>(x-1, y+1)[0]==255)&&(srcImage.at<Vec3b>(x-1, y+1)[1]==0)&&(srcImage.at<Vec3b>(x-1, y+1)[2]==0)))
				   break;
				else
				{
					ptr = traceContourSegment(srcImage, x, y, z);
					extendContourFromEndpoints++;
					stock = (int**)realloc(stock, (extendContourFromEndpoints+1)*sizeof(int*));
					stock[0][0] = extendContourFromEndpoints;
					stock[extendContourFromEndpoints] = (int*)malloc((2*ptr[0]+1)*sizeof(int));
					stock[extendContourFromEndpoints][0] = ptr[0];
					for(int k=0;k<ptr[0];k++)
					{
						stock[extendContourFromEndpoints][2*k+1] = ptr[2*k+1]; stock[extendContourFromEndpoints][2*k+2] = ptr[2*k+2];
					}
				}
			}
		}
	} //cout<<"Nm = "<<extendContourFromEndpoints<<endl<<endl;
	for(int i=1;i<I;i++)
	{
		for(int j=1;j<J;j++)
		{
			int x = i, y = j;
			if((srcImage.at<Vec3b>(x, y)[0]==255)&&(srcImage.at<Vec3b>(x, y)[1]==255)&&(srcImage.at<Vec3b>(x, y)[2]==255))
			{
				if(((srcImage.at<Vec3b>(x+1, y)[0]==255)&&(srcImage.at<Vec3b>(x+1, y)[1]==0)&&(srcImage.at<Vec3b>(x+1, y)[2]==0))||
				   ((srcImage.at<Vec3b>(x-1, y)[0]==255)&&(srcImage.at<Vec3b>(x-1, y)[1]==0)&&(srcImage.at<Vec3b>(x-1, y)[2]==0))||
				   ((srcImage.at<Vec3b>(x, y+1)[0]==255)&&(srcImage.at<Vec3b>(x, y+1)[1]==0)&&(srcImage.at<Vec3b>(x, y+1)[2]==0))||
				   ((srcImage.at<Vec3b>(x, y-1)[0]==255)&&(srcImage.at<Vec3b>(x, y-1)[1]==0)&&(srcImage.at<Vec3b>(x, y-1)[2]==0))||
				   ((srcImage.at<Vec3b>(x+1, y+1)[0]==255)&&(srcImage.at<Vec3b>(x+1, y+1)[1]==0)&&(srcImage.at<Vec3b>(x+1, y+1)[2]==0))||
				   ((srcImage.at<Vec3b>(x+1, y-1)[0]==255)&&(srcImage.at<Vec3b>(x+1, y-1)[1]==0)&&(srcImage.at<Vec3b>(x+1, y-1)[2]==0))||
				   ((srcImage.at<Vec3b>(x-1, y-1)[0]==255)&&(srcImage.at<Vec3b>(x-1, y-1)[1]==0)&&(srcImage.at<Vec3b>(x-1, y-1)[2]==0))||
				   ((srcImage.at<Vec3b>(x-1, y+1)[0]==255)&&(srcImage.at<Vec3b>(x-1, y+1)[1]==0)&&(srcImage.at<Vec3b>(x-1, y+1)[2]==0)))
				   break;
				else
				{
					ptr = traceContourSegment(srcImage, x, y, z);
					extendContourFromEndpoints++;
					stock = (int**)realloc(stock, (extendContourFromEndpoints+1)*sizeof(int*));
					stock[0][0] = extendContourFromEndpoints;
					stock[extendContourFromEndpoints] = (int*)malloc((2*ptr[0]+1)*sizeof(int));
					stock[extendContourFromEndpoints][0] = ptr[0];
					for(int k=0;k<ptr[0];k++)
					{
						stock[extendContourFromEndpoints][2*k+1] = ptr[2*k+1]; stock[extendContourFromEndpoints][2*k+2] = ptr[2*k+2];
					}
				}
			}
		}
	}

	return stock;
}

int* findGreenMarkerForGapConnection(const Mat& srcImage, int x, int y, int k)
{
	static int ptr[3] = {0, 0, 0};
	ptr[0] = 0; ptr[1] = x; ptr[2] = y;

	for(int i=1;i<=k;i++)
	{
		if((x+i>=srcImage.rows)||(x-i<=0)||(y+i>=srcImage.cols)||(y-i<=0))
			ptr[0] = 2;
		else
		{
			for(int j=-i;j<=i;j++)
			{
				if(isGreenMarker(srcImage, x+i, y+j)==1)
				{
					ptr[0] = 1;
					ptr[1] = x + i; ptr[2] = y + j;
					return ptr;
					break;
				}
				else if(isGreenMarker(srcImage, x+j, y+i)==1)
				{
					ptr[0] = 1;
					ptr[1] = x + j; ptr[2] = y + i;
					return ptr;
					break;
				}
				else if(isGreenMarker(srcImage, x+j, y-i)==1)
				{
					ptr[0] = 1;
					ptr[1] = x + j; ptr[2] = y - i;
					return ptr;
					break;
				}
				else if(isGreenMarker(srcImage, x-i, y+j)==1)
				{
					ptr[0] = 1;
					ptr[1] = x - i; ptr[2] = y + j;
					return ptr;
					break;
				}
			}
		}
	}

	return ptr;
}

void printContourSegments(int **stock)
{
	cout<<stock[0][0]<<endl<<endl;
	for(int i=1;i<=stock[0][0];i++)
	{
		cout<<i<<" extendContourFromEndpoints:"<<stock[i][0]<<" ";
		for(int j=0;j<stock[i][0];j++)
			cout<<stock[i][2*j+1]<<" "<<stock[i][2*j+2]<<"  ";
		cout<<endl;
	}
}

void printAngleProfile(double **point)
{
	cout<<point[0][0]<<endl;
	for(int i=1;i<=point[0][0];i++)
	{
		cout<<point[i][0]<<" ";
		for(int j=1;j<=point[i][0];j++)
			cout<<point[i][j]<<" ";
		cout<<endl;
	} cout<<endl;
}

int** filterContourSegments(Mat& srcImage, int **stock)
{
	//imshow("Image", srcImage);
	//printContourSegments(stock);
	for(int i=1;i<=stock[0][0];i++)
	{
		if((stock[i][0]==1)||(stock[i][0]==2)||
		  ((stock[i][0]==3)&&
		  (((abs(stock[i][1]-stock[i][3])==1)||(abs(stock[i][2]-stock[i][4])==1))||
		   ((abs(stock[i][3]-stock[i][5])==1)||(abs(stock[i][4]-stock[i][6])==1)))))
		{
			for(int j=0;j<stock[i][0];j++)
			{
				srcImage.at<Vec3b>(stock[i][2*j+1], stock[i][2*j+2])[0] = 0; srcImage.at<Vec3b>(stock[i][2*j+1], stock[i][2*j+2])[1] = 0; srcImage.at<Vec3b>(stock[i][2*j+1], stock[i][2*j+2])[2] = 0;
			}
			for(int j=0;j<(stock[0][0]-i);j++)
			{
				stock[i+j] = (int*)realloc(stock[i+j], (2*stock[i+j+1][0]+1)*sizeof(int));
				stock[i+j][0] = stock[i+j+1][0];
				for(int k=0;k<stock[i+j][0];k++)
				{
					stock[i+j][2*k+1] = stock[i+j+1][2*k+1]; stock[i+j][2*k+2] = stock[i+j+1][2*k+2];
				}
			}
			stock[0][0]--;
			stock = (int**)realloc(stock, (stock[0][0]+1)*sizeof(int*));
			i--;
		}
	}

	// printContourSegments(stock);
	for(int i=0;i<srcImage.rows;i++)
	{
		for(int j=0;j<srcImage.cols;j++)
		{
			if(((srcImage.at<Vec3b>(i, j)[0]==255)&&(srcImage.at<Vec3b>(i, j)[1]==0)&&(srcImage.at<Vec3b>(i, j)[2]==0))||
			   ((srcImage.at<Vec3b>(i, j)[0]==255)&&(srcImage.at<Vec3b>(i, j)[1]==255)&&(srcImage.at<Vec3b>(i, j)[2]==255)))
			{
				srcImage.at<Vec3b>(i, j)[0] = 0; srcImage.at<Vec3b>(i, j)[1] = 0; srcImage.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}

	for(int i=1;i<=stock[0][0];i++)
	{
		for(int j=1;j<stock[i][0];j++)
		{
			int x0 = stock[i][2*j-1], y0 = stock[i][2*j], x1 = stock[i][2*j+1], y1 = stock[i][2*j+2];
			line(srcImage, Point(y0, x0), Point(y1, x1), Scalar(255, 0, 0), 1, 8);
			srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
			srcImage.at<Vec3b>(x1, y1)[0] = 0; srcImage.at<Vec3b>(x1, y1)[1] = 0; srcImage.at<Vec3b>(x1, y1)[2] = 255;
		}
	}
	for(int i=1;i<=stock[0][0];i++)
	{
		srcImage.at<Vec3b>(stock[i][1], stock[i][2])[0] = 0; srcImage.at<Vec3b>(stock[i][1], stock[i][2])[1] = 255; srcImage.at<Vec3b>(stock[i][1], stock[i][2])[2] = 0;
		srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[0] = 144; srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[1] = 238; srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[2] = 144;
	}

	return stock;
}

int** connectContourGaps(Mat& srcImage, int **stock, int kp)
{
	int *p = nullptr;

	int Np = 0;
	for(int i=1;i<=stock[0][0];i++)
	{
		p = findGreenMarkerForGapConnection(srcImage, stock[i][1], stock[i][2], kp);  // Detect the beginning node
		if(p[0]==1)
		{
			int NI = i;
			for(int j=1;j<=stock[0][0];j++)
			{
				if((stock[j][1]==p[1])&&(stock[j][2]==p[2]))  // yellow -> yellow
				{
					int NJ = j;
					//cout<<"Y - Y"<<endl;
					//cout<<"Ny  "<<i<<" stock "<<stock[i][1]<<" "<<stock[i][2]<<endl;
					//cout<<"Ny' "<<j<<" p     "<<p[1]<<" "<<p[2]<<endl<<endl;
					line(srcImage, Point(stock[i][2], stock[i][1]), Point(p[2], p[1]), Scalar(255, 0, 0), 1, 8);
					srcImage.at<Vec3b>(stock[i][1], stock[i][2])[0] = 0; srcImage.at<Vec3b>(stock[i][1], stock[i][2])[1] = 0; srcImage.at<Vec3b>(stock[i][1], stock[i][2])[2] = 255;
					srcImage.at<Vec3b>(p[1], p[2])[0] = 0; srcImage.at<Vec3b>(p[1], p[2])[1] = 0; srcImage.at<Vec3b>(p[1], p[2])[2] = 255;
					Np++;

					int **vessel; vessel = (int**)malloc(1*sizeof(int*)); vessel[0] = (int*)malloc((2*stock[NI][0])*sizeof(int));
					for(int k=0;k<stock[NI][0];k++)
					{
						vessel[0][2*k] = stock[NI][2*stock[NI][0]-2*k-1]; vessel[0][2*k+1] = stock[NI][2*stock[NI][0]-2*k];
						//cout<<"vessel "<<vessel[0][2*k]<<" "<<vessel[0][2*k+1]<<endl<<endl;
					}
					stock[NI] = (int*)realloc(stock[NI], (2*stock[NI][0]+1+2*stock[NJ][0])*sizeof(int));
					for(int k=0;k<stock[NI][0];k++)
					{
						stock[NI][2*k+1] = vessel[0][2*k]; stock[NI][2*k+2] = vessel[0][2*k+1];
					}
					for(int k=0;k<stock[NJ][0];k++)
					{
						stock[NI][2*k+2*stock[NI][0]+1] = stock[NJ][2*k+1]; stock[NI][2*k+1+2*stock[NI][0]+1] = stock[NJ][2*k+2];
					}
					stock[NI][0] = stock[NI][0] + stock[NJ][0];
					srcImage.at<Vec3b>(stock[NI][1], stock[NI][2])[0] = 0; srcImage.at<Vec3b>(stock[NI][1], stock[NI][2])[1] = 255; srcImage.at<Vec3b>(stock[NI][1], stock[NI][2])[2] = 0;
					stock[NJ][0] = 0;
}
				if((stock[j][2*stock[j][0]-1]==p[1])&&(stock[j][2*stock[j][0]]==p[2]))  // yellow -> green
				{
					int NJ = j;
					//cout<<"Y - drawTrackedContours"<<endl;
					//cout<<"Ny  "<<i<<" stock "<<stock[i][1]<<" "<<stock[i][2]<<endl;
					//cout<<"Ny' "<<j<<" p     "<<p[1]<<" "<<p[2]<<endl<<endl;
					line(srcImage, Point(stock[i][2], stock[i][1]), Point(p[2], p[1]), Scalar(255, 0, 0), 1, 8);
					srcImage.at<Vec3b>(stock[i][1], stock[i][2])[0] = 0; srcImage.at<Vec3b>(stock[i][1], stock[i][2])[1] = 0; srcImage.at<Vec3b>(stock[i][1], stock[i][2])[2] = 255;
					srcImage.at<Vec3b>(p[1], p[2])[0] = 0; srcImage.at<Vec3b>(p[1], p[2])[1] = 0; srcImage.at<Vec3b>(p[1], p[2])[2] = 255;
					Np++;

					int **vessel; vessel = (int**)malloc(1*sizeof(int*)); vessel[0] = (int*)malloc((2*stock[NI][0])*sizeof(int));
					for(int k=0;k<stock[NI][0];k++)
					{
						vessel[0][2*k] = stock[NI][2*k+1]; vessel[0][2*k+1] = stock[NI][2*k+2];
						//cout<<"vessel "<<vessel[0][2*k]<<" "<<vessel[0][2*k+1]<<endl<<endl;
					}
					stock[NJ] = (int*)realloc(stock[NJ], (2*stock[NI][0]+1+2*stock[NJ][0])*sizeof(int));
					for(int k=0;k<stock[NI][0];k++)
					{
						stock[NJ][2*k+2*stock[NJ][0]+1] = vessel[0][2*k]; stock[NJ][2*k+1+2*stock[NJ][0]+1] = vessel[0][2*k+1];
					}
					stock[NJ][0] = stock[NI][0] + stock[NJ][0];
					stock[NI][0] = 0;
}
			}
		}

		p = findGreenMarkerForGapConnection(srcImage, stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]], kp);  // Detect the terminal node
		if(p[0]==1)
		{
			int NI = i;
			for(int j=1;j<=stock[0][0];j++)
			{
				if((stock[j][1]==p[1])&&(stock[j][2]==p[2]))  // green -> yellow
				{
					int NJ = j;
					//cout<<"drawTrackedContours - Y"<<endl;
					//cout<<"Ng  "<<i<<" stock "<<stock[i][2*stock[i][0]-1]<<" "<<stock[i][2*stock[i][0]]<<endl;
					//cout<<"Ng' "<<j<<" p     "<<p[1]<<" "<<p[2]<<endl<<endl;
					line(srcImage, Point(stock[i][2*stock[i][0]], stock[i][2*stock[i][0]-1]), Point(p[2], p[1]), Scalar(255, 0, 0), 1, 8);
					srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[0] = 0; srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[1] = 0; srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[2] = 255;
					srcImage.at<Vec3b>(p[1], p[2])[0] = 0; srcImage.at<Vec3b>(p[1], p[2])[1] = 0; srcImage.at<Vec3b>(p[1], p[2])[2] = 255;
					Np++;

					int **vessel; vessel = (int**)malloc(1*sizeof(int*)); vessel[0] = (int*)malloc((2*stock[NJ][0])*sizeof(int));
					for(int k=0;k<stock[NJ][0];k++)
					{
						vessel[0][2*k] = stock[NJ][2*k+1]; vessel[0][2*k+1] = stock[NJ][2*k+2];
						//cout<<"vessel "<<vessel[0][2*k]<<" "<<vessel[0][2*k+1]<<endl<<endl;
					}
					stock[NI] = (int*)realloc(stock[NI], (2*stock[NI][0]+1+2*stock[NJ][0])*sizeof(int));
					for(int k=0;k<stock[NJ][0];k++)
					{
						stock[NI][2*k+2*stock[NI][0]+1] = vessel[0][2*k]; stock[NI][2*k+1+2*stock[NI][0]+1] = vessel[0][2*k+1];
					}
					stock[NI][0] = stock[NI][0] + stock[NJ][0];
					stock[NJ][0] = 0;
}
				if((stock[j][2*stock[j][0]-1]==p[1])&&(stock[j][2*stock[j][0]]==p[2]))  // green -> green
				{
					int NJ = j;
					//cout<<"drawTrackedContours - drawTrackedContours"<<endl;
					//cout<<"Ng  "<<i<<" stock "<<stock[i][2*stock[i][0]-1]<<" "<<stock[i][2*stock[i][0]]<<endl;
					//cout<<"Ng' "<<j<<" p     "<<p[1]<<" "<<p[2]<<endl<<endl;
					line(srcImage, Point(stock[i][2*stock[i][0]], stock[i][2*stock[i][0]-1]), Point(p[2], p[1]), Scalar(255, 0, 0), 1, 8);
					srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[0] = 0; srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[1] = 0; srcImage.at<Vec3b>(stock[i][2*stock[i][0]-1], stock[i][2*stock[i][0]])[2] = 255;
					srcImage.at<Vec3b>(p[1], p[2])[0] = 0; srcImage.at<Vec3b>(p[1], p[2])[1] = 0; srcImage.at<Vec3b>(p[1], p[2])[2] = 255;
					Np++;

					int **vessel; vessel = (int**)malloc(1*sizeof(int*)); vessel[0] = (int*)malloc((2*stock[NI][0])*sizeof(int));
					for(int k=0;k<stock[NI][0];k++)
					{
						vessel[0][2*k] = stock[NI][2*stock[NI][0]-2*k-1]; vessel[0][2*k+1] = stock[NI][2*stock[NI][0]-2*k];
						//cout<<"vessel "<<vessel[0][2*k]<<" "<<vessel[0][2*k+1]<<endl<<endl;
					}
					stock[NJ] = (int*)realloc(stock[NJ], (2*stock[NI][0]+1+2*stock[NJ][0])*sizeof(int));
					for(int k=0;k<stock[NI][0];k++)
					{
						stock[NJ][2*k+2*stock[NJ][0]+1] = vessel[0][2*k]; stock[NJ][2*k+1+2*stock[NJ][0]+1] = vessel[0][2*k+1];
					}
					stock[NJ][0] = stock[NI][0] + stock[NJ][0];
					srcImage.at<Vec3b>(stock[NJ][2*stock[NJ][0]-1],stock[NJ][2*stock[NJ][0]])[0] = 144; srcImage.at<Vec3b>(stock[NJ][2*stock[NJ][0]-1],stock[NJ][2*stock[NJ][0]])[1] = 238; srcImage.at<Vec3b>(stock[NJ][2*stock[NJ][0]-1],stock[NJ][2*stock[NJ][0]])[2] = 144;
					stock[NI][0] = 0;
}
			}
		}
	}
	//cout<<"Np = "<<Np<<endl;

	for(int i=1;i<=stock[0][0];i++)
	{
		if(stock[i][0]==0)
		{
			for(int j=0;j<(stock[0][0]-i);j++)
			{
				stock[i+j] = (int*)realloc(stock[i+j], (2*stock[i+j+1][0]+1)*sizeof(int));
				stock[i+j][0] = stock[i+j+1][0];
				for(int k=0;k<stock[i+j][0];k++)
				{
					stock[i+j][2*k+1] = stock[i+j+1][2*k+1]; stock[i+j][2*k+2] = stock[i+j+1][2*k+2];
				}
			}
			stock[0][0]--;
			stock = (int**)realloc(stock, (stock[0][0]+1)*sizeof(int*));
			i--;
		}
	}

	// printContourSegments(stock);

	return stock;
}

double** calculateContourAngles(const Mat& srcImage, int **stock)
{
	double **point;
	point = (double**)malloc((stock[0][0]+1)*sizeof(double*));
	point[0] = (double*)malloc(1*sizeof(double));
	point[0][0] = stock[0][0];
	for(int i=1;i<=stock[0][0];i++)
		point[i] = (double*)malloc((stock[i][0]+1)*sizeof(double));

	double calculateContourAngles; double k; int dx, dy;
	for(int i=1;i<=stock[0][0];i++)
	{
		point[i][0] = stock[i][0]-1;
		for(int j=0;j<point[i][0];j++)
		{
			dx = stock[i][2*j+3] - stock[i][2*j+1]; dy = stock[i][2*j+4] - stock[i][2*j+2];
			if((dx!=0)&&(dy!=0))
			{
				k = 1.0 * dy / dx;
				calculateContourAngles = atan2(k, 1)*180/3.14159;
				if(calculateContourAngles<0)
					calculateContourAngles += 180;
				if(((k>0)&&((dx<0)||(dy<0)))||((k<0)&&((dx>0)||(dy<0))))
					calculateContourAngles += 180;

				point[i][j+1] = calculateContourAngles;
			}
			else if((dx>0)&&(dy==0))  // down
				point[i][j+1] = 0;
			else if((dx<0)&&(dy==0))  // up
				point[i][j+1] = 180;
			else if((dx==0)&&(dy>0))  // right
				point[i][j+1] = 90;
			else if((dx==0)&&(dy<0))  // left
				point[i][j+1] = 270;
		}
	}

	// printAngleProfile(point);

	return point;
}

double** calculateAngleChanges(const Mat& srcImage, double **stock)
{
	double **point;
	point = (double**)malloc((stock[0][0]+1)*sizeof(double*));
	point[0] = (double*)malloc(1*sizeof(double));
	point[0][0] = stock[0][0];
	for(int i=1;i<=stock[0][0];i++)
		point[i] = (double*)malloc((stock[i][0]+1)*sizeof(double));

	double calculateAngleChanges;

	for(int i=1;i<=point[0][0];i++)
	{
		point[i][0] = stock[i][0]-1;
		for(int j=1;j<=point[i][0];j++)
		{
			calculateAngleChanges = stock[i][j+1] - stock[i][j];
			if(calculateAngleChanges>180)
				calculateAngleChanges -= 360;
			else if(calculateAngleChanges<-180)
				calculateAngleChanges += 360;

			point[i][j] = calculateAngleChanges;
		}
	}

	//cout<<"calculateAngleChanges**************************************"<<endl;
	//printAngleProfile(point);
	/*
	cout<<"S**************************************"<<endl;
	cout<<point[0][0]<<endl;
	for(int i=1;i<=point[0][0];i++)
	{
		cout<<point[i][0]<<" ";
		for(int j=1;j<=point[i][0];j++)
			cout<<(int)point[i][j]/10<<" ";
		cout<<endl;
	} cout<<endl;
	*/

	return point;
}

int* markHighCurvaturePoints(Mat& srcImage, int **stock, double **point_dTheta)
{
	int *point; point = (int*)malloc(1*sizeof(int)); point[0] = 0;
	int k = 0;

	for(int i=1;i<=point_dTheta[0][0];i++)
	{
		for(int j=1;j<=point_dTheta[i][0];j++)
		{
			//cout<<(int)point_dTheta[i][j]/10<<" ";
			if(abs((int)point_dTheta[i][j]/10)>=5)
			{
				point = (int*)realloc(point, (2*point[0]+3)*sizeof(int));
				point[2*k+1] = i; point[2*k+2] = j + 1; point[0] += 1;
				k++;
				srcImage.at<Vec3b>(stock[i][2*j+1], stock[i][2*j+2])[0] = 0; srcImage.at<Vec3b>(stock[i][2*j+1], stock[i][2*j+2])[1] = 255; srcImage.at<Vec3b>(stock[i][2*j+1], stock[i][2*j+2])[2] = 0;
			}
		} //cout<<endl;
	}

	/*
	cout<<"markHighCurvaturePoints***********************************************"<<endl;
	cout<<point[0]<<endl;
	for(int i=1;i<=2*point[0];i++)
		cout<<point[i]<<" ";
	cout<<endl;
	*/

	return point;
}

int** extendContourFromEndpoints(Mat& srcImage, int **stock, double **point_Theta, int K)
{
	int *ptr = nullptr;
	int **point; point = (int**)malloc((stock[0][0]+1)*sizeof(int*));
	point[0] = (int*)malloc(1*sizeof(int)); point[0][0] = stock[0][0];
	for(int i=1;i<=stock[0][0];i++)
	{
		point[i] = (int*)malloc(3*sizeof(int));
		point[i][0] = 1;
		if(K==0)
		{ point[i][1] = stock[i][1]; point[i][2] = stock[i][2]; }
		else if(K==1)
		{ point[i][1] = stock[i][2*stock[i][0]-1]; point[i][2] = stock[i][2*stock[i][0]]; }
	}

	double calculateContourAngles, calculateAngleChanges;
	int l = 20, m = 50, limit = 80;  // l - bound scope, m - range of lengths, limit - range of angles

	for(int i=1;i<=stock[0][0];i++)
	{//int i = 1;
		int dTh = 0;
		//if(stock[i][0]>=3)
		//{ m = 50; limit = 80; }
		//else
		//{ m = 50; limit = 50; }

		for(int d=dTh-limit;d<=dTh+limit;d++)
		{//int d = 0;
			if(K==0)
				calculateContourAngles = d + point_Theta[i][1];
			else if(K==1)
				calculateContourAngles = d + point_Theta[i][(int)point_Theta[i][0]];
			//cout<<"calculateContourAngles:"<<calculateContourAngles<<endl;
			int x, y; double k, b; int x0, y0, xt, yt;
			if(K==0)
			{
				x0 = stock[i][1]; y0 = stock[i][2];
				xt = x0; yt = y0; //cout<<"x0 = "<<x0<<" y0 = "<<y0<<endl;
			}
			else if(K==1)
			{
				x0 = stock[i][2*stock[i][0]-1]; y0 = stock[i][2*stock[i][0]];
				xt = x0; yt = y0; //cout<<"x0 = "<<x0<<" y0 = "<<y0<<endl;
			}

			int dx, dy;
			if(K==0)
			{
				dx = stock[i][1] - stock[i][3];
				dy = stock[i][2] - stock[i][4];
			}
			else if(K==1)
			{
				dx = stock[i][2*stock[i][0]-1] - stock[i][2*stock[i][0]-3];
				dy = stock[i][2*stock[i][0]] - stock[i][2*stock[i][0]-2];
			}
			//cout<<"calculateContourAngles = "<<calculateContourAngles<<endl;

			k = tan(calculateContourAngles/180*3.14159);
			if(k>srcImage.cols-2)
				k = srcImage.cols-2;
			else if(k<-srcImage.cols+2)
				k = -srcImage.cols+2;
			b = y0 - k * x0; //cout<<"d:"<<d<<" kt = "<<k<<" bt = "<<b<<endl;

			x = x0; y = y0;
			srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;

			if((x0<srcImage.rows-l)&&(y0<srcImage.cols-l)&&(x0>l)&&(y0>l))
			{
				if(((dx<0)||(dy>0))&&(k<0))
				{
					do
					{
						if(abs(k)<1)
						{
							x--; y = k * x + b;
						}
						else if(abs(k)>=1)
						{
							y++; x = (y - b) / k;
						}
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}
				else if(((dx<0)||(dy<0))&&(k>0))
				{
					do
					{
						if(abs(k)<1)
						{
							x--; y = k * x + b;
						}
						else if(abs(k)>=1)
						{
							y--; x = (y - b) / k;
						}
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}
				if(((dx>0)||(dy>0))&&(k>0))
				{
					do
					{
						if(abs(k)<1)
						{
							x++; y = k * x + b;
						}
						else if(abs(k)>=1)
						{
							y++; x = (y - b) / k;
						}
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}
				else if(((dx>0)||(dy<0))&&(k<0))
				{
					do
					{
						if(abs(k)<1)
						{
							x++; y = k * x + b;
						}
						else if(abs(k)>=1)
						{
							y--; x = (y - b) / k;
						}
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}
				else if((dx==0)&&(dy>0))
				{
					do
					{
						y++;
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}
				else if((dx==0)&&(dy<0))
				{
					do
					{
						y--;
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}
				else if((dx>0)&&(dy==0))
				{
					do
					{
						x++;
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}
				else if((dx<0)&&(dy==0))
				{
					do
					{
						x--;
						ptr = findGreenMarkerNearby(srcImage, x, y, 2);
						if(ptr[0]==1)
						{
							xt = ptr[1]; yt = ptr[2];
							if(((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt))&&
							   ((point[i][2*point[i][0]-1]!=xt)&&(point[i][2*point[i][0]]!=yt)))
							{
								point[i] = (int*)realloc(point[i], (2*point[i][0]+3)*sizeof(int)); point[i][0] += 1;
								point[i][2*point[i][0]-1] = xt; point[i][2*point[i][0]] = yt;
							}
							line(srcImage, Point(y0, x0), Point(yt, xt), Scalar(255, 255, 0), 1, 8);
							srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 0; srcImage.at<Vec3b>(x0, y0)[2] = 255;
							srcImage.at<Vec3b>(xt, yt)[0] = 0; srcImage.at<Vec3b>(xt, yt)[1] = 255; srcImage.at<Vec3b>(xt, yt)[2] = 0;
							break;
						}
						if((sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))>m)||(isBlueOrRedMarker(srcImage, x, y)==1))
							break;
					} while((x>l)&&(y>l)&&(x<srcImage.rows-l)&&(y<srcImage.cols-l));
				}

				srcImage.at<Vec3b>(x0, y0)[0] = 0; srcImage.at<Vec3b>(x0, y0)[1] = 255; srcImage.at<Vec3b>(x0, y0)[2] = 0;
			}
		}
	}

	//cout<<"Nn*****************************************"<<endl;
	//printContourSegments(point);

	return point;
}

int** deduplicateTracePoints(int **stock, int **point_k, int *point_o)
{
	int x, y;

	for(int i=1;i<=point_k[0][0];i++)
	{
		for(int j=1;j<=point_k[i][0];j++)
		{
			x = point_k[i][2*j-1]; y = point_k[i][2*j];
			for(int k=j;k<point_k[i][0];k++)
			{
				if((point_k[i][2*k+1]==x)&&(point_k[i][2*k+2]==y))
				{
					for(int l=0;l<point_k[i][0]-k;l++)
					{ point_k[i][2*k+2*l+1] = point_k[i][2*k+2*l+3]; point_k[i][2*k+2*l+2] = point_k[i][2*k+2*l+4]; }
					point_k[i][0] -= 1; point_k[i] = (int*)realloc(point_k[i], (2*point_k[i][0]+1)*sizeof(int));
					k--;
				}
			}
		}
	}

	int xt, yt;
	for(int i=1;i<=point_k[0][0];i++)
	{
		for(int j=0;j<point_k[i][0];j++)
		{
			xt = point_k[i][2*j+1]; yt = point_k[i][2*j+2];
			for(int k=1;k<=stock[0][0];k++)
			{
				if((xt==stock[k][1])&&(yt==stock[k][2]))
				{ point_k[i][2*j+1] = k; point_k[i][2*j+2] = 1; }
				else if((xt==stock[k][2*stock[k][0]-1])&&(yt==stock[k][2*stock[k][0]]))
				{ point_k[i][2*j+1] = k; point_k[i][2*j+2] = stock[k][0]; }
				else
				{
					for(int l=1;l<=point_o[0];l++)
					{
						int oi = point_o[2*l-1], oj = point_o[2*l];
						if((xt==stock[oi][2*oj-1])&&(yt==stock[oi][2*oj]))
						{ point_k[i][2*j+1] = oi; point_k[i][2*j+2] = oj; }
					}
				}
			}
		}
	}

	//cout<<"deduplicateTracePoints****************************************"<<endl;
	//printContourSegments(point_k);

	return point_k;
}

int** mergeEndpointTraces(int **point_k1, int **point_k2)
{
	int **point_k; point_k = (int**)malloc((point_k1[0][0]+point_k2[0][0]+1)*sizeof(int*));
	point_k[0] = (int*)malloc(1*sizeof(int)); point_k[0][0] = point_k1[0][0];

	for(int i=1;i<=point_k[0][0];i++)
	{
		point_k[2*i-1] = (int*)malloc((2*point_k1[i][0]+1)*sizeof(int)); point_k[2*i-1][0] = point_k1[i][0];
		for(int j=0;j<point_k1[i][0];j++)
		{ point_k[2*i-1][2*j+1] = point_k1[i][2*j+1]; point_k[2*i-1][2*j+2] = point_k1[i][2*j+2]; }

		point_k[2*i] = (int*)malloc((2*point_k2[i][0]+1)*sizeof(int)); point_k[2*i][0] = point_k2[i][0];
		for(int j=0;j<point_k2[i][0];j++)
		{ point_k[2*i][2*j+1] = point_k2[i][2*j+1]; point_k[2*i][2*j+2] = point_k2[i][2*j+2]; }
	} point_k[0][0] *= 2;

	return point_k;
}

void drawTrackedContours(Mat& srcImage, int **stock, int **point_k)
{
	int x0, y0, xt, yt; int x1, y1;
	
	for(int i=0;i<srcImage.rows;i++)
	{
		for(int j=0;j<srcImage.cols;j++)
		{
			if(((srcImage.at<Vec3b>(i, j)[0]==255)&&(srcImage.at<Vec3b>(i, j)[1]==0)&&(srcImage.at<Vec3b>(i, j)[2]==0))||
			  ((srcImage.at<Vec3b>(i, j)[0]==0)&&(srcImage.at<Vec3b>(i, j)[1]==0)&&(srcImage.at<Vec3b>(i, j)[2]==255))||
			  ((srcImage.at<Vec3b>(i, j)[0]==0)&&(srcImage.at<Vec3b>(i, j)[1]==255)&&(srcImage.at<Vec3b>(i, j)[2]==0))||
			  ((srcImage.at<Vec3b>(i, j)[0]==144)&&(srcImage.at<Vec3b>(i, j)[1]==238)&&(srcImage.at<Vec3b>(i, j)[2]==144))||
			  ((srcImage.at<Vec3b>(i, j)[0]==255)&&(srcImage.at<Vec3b>(i, j)[1]==255)&&(srcImage.at<Vec3b>(i, j)[2]==0)))
			{ srcImage.at<Vec3b>(i, j)[0] = 255; srcImage.at<Vec3b>(i, j)[1] = 255; srcImage.at<Vec3b>(i, j)[2] = 255; }
		}
	}
	
	//cout<<"detectBrightContours********"<<endl; printContourSegments(point_k);
}

void addDetectionBorder(Mat& srcImage, int k)
{
	int a, b;

	b = k;
	for(a=k;a<(srcImage.rows-k)/2;a++)
	{
		if((srcImage.at<Vec3b>(a, b)[0]==255)&&(srcImage.at<Vec3b>(a, b)[1]==255)&&(srcImage.at<Vec3b>(a, b)[1]==255))
			break;
		srcImage.at<Vec3b>(a, b)[0] = 255; srcImage.at<Vec3b>(a, b)[1] = 255; srcImage.at<Vec3b>(a, b)[2] = 255;
	}

	b = srcImage.cols - k;
	for(a=k;a<(srcImage.rows-k)/2;a++)
	{
		if((srcImage.at<Vec3b>(a, b)[0]==255)&&(srcImage.at<Vec3b>(a, b)[1]==255)&&(srcImage.at<Vec3b>(a, b)[1]==255))
			break;
		srcImage.at<Vec3b>(a, b)[0] = 255; srcImage.at<Vec3b>(a, b)[1] = 255; srcImage.at<Vec3b>(a, b)[2] = 255;
	}

	a = k;
	for(b=k;b<srcImage.cols-k;b++)
	{ srcImage.at<Vec3b>(a, b)[0] = 255; srcImage.at<Vec3b>(a, b)[1] = 255; srcImage.at<Vec3b>(a, b)[2] = 255; }

}

void removeDetectionBorder(Mat& srcImage, int k)
{
	int a, b;

	b = k;
	for(a=k;a<(srcImage.rows-k)/2;a++)
	{
		if((srcImage.at<Vec3b>(a, b)[0]==0)&&(srcImage.at<Vec3b>(a, b)[1]==0)&&(srcImage.at<Vec3b>(a, b)[1]==0))
			break;
		srcImage.at<Vec3b>(a, b)[0] = 0; srcImage.at<Vec3b>(a, b)[1] = 0; srcImage.at<Vec3b>(a, b)[2] = 0;
	}

	b = srcImage.cols - k;
	for(a=k;a<(srcImage.rows-k)/2;a++)
	{
		if((srcImage.at<Vec3b>(a, b)[0]==0)&&(srcImage.at<Vec3b>(a, b)[1]==0)&&(srcImage.at<Vec3b>(a, b)[1]==0))
			break;
		srcImage.at<Vec3b>(a, b)[0] = 0; srcImage.at<Vec3b>(a, b)[1] = 0; srcImage.at<Vec3b>(a, b)[2] = 0;
	}

	a = k;
	for(b=k;b<srcImage.cols-k;b++)
	{ srcImage.at<Vec3b>(a, b)[0] = 0; srcImage.at<Vec3b>(a, b)[1] = 0; srcImage.at<Vec3b>(a, b)[2] = 0; }

}

int* estimateDropletBoundaries(Mat& srcImage, int **stock, double **point_dTheta)
{
	int *point; point = (int*)malloc(5*sizeof(int));
	int limit = 25; int kl1 = 25, kl2 = 30;
	int* s; s = (int*)malloc(stock[0][0]*sizeof(int));
	for(int i=0;i<stock[0][0];i++)
	{ s[i] = stock[i+1][0]; }

	//line(srcImage, Point(0, 100), Point(srcImage.cols, 100), Scalar(255, 255, 255), 1, 8);
	//line(srcImage, Point(0, 100+limit), Point(srcImage.cols, 100+limit), Scalar(255, 255, 255), 1, 8);

	int temp;
    for(int i=0;i<stock[0][0];i++)
    {
        for(int j=i+1;j<stock[0][0];j++)
        {
            if(s[i]<s[j])
				swap(s[i], s[j]);
        }
    }

	if(s[0]>=55)
	{
		int I;
		for(int i=1;i<=stock[0][0];i++)  // Calculate the upper limit
		{
			if(s[0]==stock[i][0])
			{ I = i; break; }
		}
		int *point_s;
		//cout<<"POINT_S*********"<<endl;
		point_s = (int*)malloc((point_dTheta[I][0]+1)*sizeof(int));
		point_s[0] = point_dTheta[I][0]-(kl1+kl2-1); //cout<<point_s[0]<<endl;
		int j = 1;
		for(int i=kl1;i<=point_dTheta[I][0]-kl2;i++)
		{
			point_s[j] = (int)point_dTheta[I][i]/10;
			srcImage.at<Vec3b>(stock[I][2*i+1], stock[I][2*i+2])[0]=255; srcImage.at<Vec3b>(stock[I][2*i+1], stock[I][2*i+2])[1]=255; srcImage.at<Vec3b>(stock[I][2*i+1], stock[I][2*i+2])[2]=255;
			j++;
		}
		/*
		for(int i=1;i<=point_s[0];i++)
			cout<<point_s[i]<<" ";
		cout<<endl;
		*/
		int I1, I2;
		for(int i=1;i<=point_s[0];i++)
		{
			if(point_s[i]!=0)
			{ I1 = i; break; }
		}
		for(int i=point_s[0];i>=1;i--)
		{
			if(point_s[i]!=0)
			{ I2 = i; break; }
		}
		//cout<<" I1 = "<<I1<<" I2 = "<<I2<<endl;

		point[0] = stock[I][2*I1+2*kl1-1]; point[1] = stock[I][2*I1+2*kl1]; point[2] = stock[I][2*I2+2*kl1-1]; point[3] = stock[I][2*I2+2*kl1];
		if(abs(point[0]-point[2])>=limit)
		{
			if(point[0]>point[2])
			{ point[0] = stock[I][2*I1+2*kl1-5]; point[1] = stock[I][2*I1+2*kl1-4]; }
			else
			{ point[2] = stock[I][2*I2+2*kl1+3]; point[3] = stock[I][2*I2+2*kl1+4]; }
		}
		line(srcImage, Point(point[1], point[0]), Point(point[3], point[2]), Scalar(0, 0, 255), 1, 8);  // Draw the upper limit

		int Jxmax = 0, Xmax = 0;  // Calculate the lower limit

		for(int j=0;j<stock[I][0]-1;j++)
		{
			if(stock[I][2*j+1]>Xmax)
			{ Xmax = stock[I][2*j+1]; Jxmax = j; }
			//cout<<"Jxmax = "<<Jxmax<<" Xmax = "<<Xmax<<endl;
		} Xmax = stock[I][2*Jxmax+1];
		//cout<<"Xmax = "<<Xmax<<endl;

		point[4] = Xmax;
		line(srcImage, Point(0, point[4]), Point(srcImage.cols, point[4]), Scalar(0, 255, 0), 1, 8);  // Draw the lower limit

		return point;
	}
	else if(s[0]<55)
	{
		for(int i=1;i<=stock[0][0];i++)
		{
			if(stock[i][2]>stock[i][2*stock[i][0]])
			{
				int *vessel; vessel = (int*)malloc(2*stock[i][0]*sizeof(int));
				for(int j=0;j<stock[i][0];j++)
				{ vessel[2*stock[i][0]-2*j-1] = stock[i][2*j+1]; vessel[2*stock[i][0]-2*j] = stock[i][2*j+2]; }
				for(int j=0;j<stock[i][0];j++)
				{ stock[i][2*j+1] = vessel[2*j+1]; stock[i][2*j+2] = vessel[2*j+2]; }
			}
		}

		int i1 = 0, i2 = 0;
		for(int i=1;i<=stock[0][0];i++)
		{
			if((stock[i][1]<=100))
				i1 = i;
			if(stock[i][2*stock[i][0]-1]<=100)
				i2 = i;
		} //cout<<" i1 = "<<i1<<" i2 = "<<i2<<endl;

		int s0, s1;
		if(stock[i1][0]>2*kl1)
			s0 = stock[i1][0]*2/5;
		else
			s0 = stock[i1][0]*1/4;
		if(stock[i2][0]>2*kl2)
			s1 = stock[i2][0]*2/5;
		else
			s1 = stock[i2][0]*1/4;
		//cout<<"s0 = "<<s0<<" s1 = "<<s1<<endl;

		int I1, I2;
		int *point_s1, *point_s2;
		point_s1 = (int*)malloc((s[0]*2/5+1)*sizeof(int)); point_s2 = (int*)malloc((s[0]*2/5+1)*sizeof(int));
		point_s1[0] = s0; point_s2[0] = s1;

		int j1 = 1, j2 = 1;
		for(int i=point_dTheta[i1][0]-s0;i<=point_dTheta[i1][0];i++)  // Calculate the upper limit
		{
			point_s1[j1] = (int)point_dTheta[i1][i]/10;
			srcImage.at<Vec3b>(stock[i1][2*i+1], stock[i1][2*i+2])[0]=255; srcImage.at<Vec3b>(stock[i1][2*i+1], stock[i1][2*i+2])[1]=255; srcImage.at<Vec3b>(stock[i1][2*i+1], stock[i1][2*i+2])[2]=255;
			j1++;
		}
		for(int i=s1;i>=1;i--)
		{
			point_s2[j2] = (int)point_dTheta[i2][i]/10;
			srcImage.at<Vec3b>(stock[i2][2*i+1], stock[i2][2*i+2])[0]=255; srcImage.at<Vec3b>(stock[i2][2*i+1], stock[i2][2*i+2])[1]=255; srcImage.at<Vec3b>(stock[i2][2*i+1], stock[i2][2*i+2])[2]=255;
			j2++;
		}

		for(int i=1;i<=point_s1[0];i++)
		{
			if(point_s1[i]!=0)
			{ I1 = i; break; }
		}
		for(int i=1;i<=point_s2[0];i++)
		{
			if(point_s2[i]!=0)
			{ I2 = s1 + 1 - i; break; }
		}
		//cout<<" I1 = "<<I1<<" I2 = "<<I2<<endl;

		point[0] = stock[i1][2*(I1+stock[i1][0]-s0)-5]; point[1] = stock[i1][2*(I1+stock[i1][0]-s0)-4]; point[2] = stock[i2][2*I2+3]; point[3] = stock[i2][2*I2+4];
		if(abs(point[0]-point[2])>=limit)
		{
			if(point[0]>point[2])
			{ point[0] = stock[i1][2*(I1+stock[i1][0]-s0)-9]; point[1] = stock[i1][2*(I1+stock[i1][0]-s0)-8]; }
			else
			{ point[2] = stock[i2][2*I2-1];  point[2] = stock[i2][2*I2]; }
		}
		line(srcImage, Point(point[1], point[0]), Point(point[3], point[2]), Scalar(0, 0, 255), 1, 8);  // Draw the upper limit


		int Jxmax1 = 0, Xmax1 = 0;  // Calculate the lower limit
		for(int j=0;j<stock[i1][0]-1;j++)
		{
			if(stock[i1][2*j+1]>Xmax1)
			{ Xmax1 = stock[i1][2*j+1]; Jxmax1 = j; }
			//cout<<"Jxmax1 = "<<Jxmax1<<" Xmax1 = "<<Xmax1<<endl;
		} Xmax1 = stock[i1][2*Jxmax1+1];
		//cout<<"Xmax1 = "<<Xmax1<<endl;

		int Jxmax2 = 0, Xmax2 = 0;  // Calculate the lower limit
		for(int j=0;j<stock[i2][0]-1;j++)
		{
			if(stock[i2][2*j+1]>Xmax2)
			{ Xmax2 = stock[i2][2*j+1]; Jxmax2 = j; }
			//cout<<"Jxmax2 = "<<Jxmax2<<" Xmax2 = "<<Xmax2<<endl;
		} Xmax2 = stock[i2][2*Jxmax2+1];
		//cout<<"Xmax2 = "<<Xmax2<<endl;

		int Xmax, Yxmax;
		if(Xmax1>=Xmax2)
		{ Xmax = Xmax1; }
		else
		{ Xmax = Xmax2; }

		point[4] = Xmax;
		line(srcImage, Point(0, point[4]), Point(srcImage.cols, point[4]), Scalar(0, 255, 0), 1, 8);  // Draw the lower limit

		return point;
	}
}

void detectBrightContours(Mat& srcImage)
{
	AllocationScope allocationScope;
	int **stock;
	double **point_Theta;
	double **point_dTheta;
	int *point_o;
	int **point_k1, **point_k2, **point_k;

	stock = extractContourSegments(srcImage, 8);
	stock = filterContourSegments(srcImage, stock); //cout<<"filterContourSegments********"<<endl; printContourSegments(stock);
	stock = connectContourGaps(srcImage, stock, 3); //cout<<"connectContourGaps********"<<endl; printContourSegments(stock);
	point_Theta = calculateContourAngles(srcImage, stock);
	point_dTheta = calculateAngleChanges(srcImage, point_Theta);

	point_o = markHighCurvaturePoints(srcImage, stock, point_dTheta);
	point_k1 = extendContourFromEndpoints(srcImage, stock, point_Theta, 0); //cout<<"N1********"<<endl; printContourSegments(point_k1);
	point_k2 = extendContourFromEndpoints(srcImage, stock, point_Theta, 1); //cout<<"N2********"<<endl; printContourSegments(point_k2);

	point_k1 = deduplicateTracePoints(stock, point_k1, point_o); //cout<<"M1********"<<endl; printContourSegments(point_k1);
	point_k2 = deduplicateTracePoints(stock, point_k2, point_o); //cout<<"M2********"<<endl; printContourSegments(point_k2);

	point_k = mergeEndpointTraces(point_k1, point_k2); //cout<<"K********"<<endl; printContourSegments(point_k);

	drawTrackedContours(srcImage, stock, point_k);
}

std::array<int, 5> trackDropletBoundary(Mat& srcImage)
{
	AllocationScope allocationScope;
	int **stock;
	int *point_l;
	double **point_Theta;
	double **point_dTheta;

	stock = extractContourSegments(srcImage, 8);
	stock = filterContourSegments(srcImage, stock); //cout<<"filterContourSegments********"<<endl; printContourSegments(stock);
	stock = connectContourGaps(srcImage, stock, 3); //cout<<"connectContourGaps********"<<endl; printContourSegments(stock);

	point_Theta = calculateContourAngles(srcImage, stock); //cout<<"calculateContourAngles********"<<endl; printAngleProfile(point_Theta);
	point_dTheta = calculateAngleChanges(srcImage, point_Theta); //cout<<"calculateAngleChanges********"<<endl; printAngleProfile(point_dTheta);

	point_l = estimateDropletBoundaries(srcImage, stock, point_dTheta);

	std::array<int, 5> point{};
	for(int i=0;i<5;i++)
		point[i] = point_l[i];

	return point;
}

int main()
{
	Mat srcImage, srcImage2, srcImage3; std::array<int, 5> point_E{};

	ofstream outfile;
    outfile.open("welding-droplet-vision-data\\outputs-traditional_vision\\24c.csv", ios::out | ios::trunc);
    outfile<<"upper_left_y"<<","<<"upper_left_x"<<","<<"upper_right_y"<<","<<"upper_right_x"<<","<<"bottom_y"<<endl;

	// for(int k=482;k<=1036;k++)
	{int k = 2;
		stringstream str1, str2, str3;

        str1 << "welding-droplet-vision-data\\frames\\24c\\" << k << ".png";
        str3 << "welding-droplet-vision-data\\outputs-traditional_vision\\24c\\" << k << ".png";

		srcImage = imread(str1.str());
		srcImage2 = imread(str1.str());

		GaussianBlur(srcImage, srcImage, Size(5, 5), 0, 0, BORDER_DEFAULT);
		int edgeThresh = 40;
		Mat edgeImage;
		// OpenCV 5 forbids in-place Canny when the source is not CV_8UC1.
		// Keep the original three-channel input and write edges to a separate Mat.
		Canny(srcImage, edgeImage, edgeThresh, edgeThresh*0.5, 3);
		cvtColor(edgeImage, srcImage, COLOR_GRAY2BGR);

		maskBelowRow(srcImage, 260);

		detectBrightContours(srcImage);

		addDetectionBorder(srcImage, 10);

		cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
		vector<vector<Point>> contours_d;
		vector<Vec4i> hierarchy_d;
		findContours(srcImage, contours_d, hierarchy_d, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		for (int i=0;i<(int)contours_d.size();i++)
			drawContours(srcImage, contours_d, i, Scalar(255), -1, 8);

		vector<vector<Point>> contours_c;
		vector<Vec4i> hierarchy_c;
		findContours(srcImage, contours_c, hierarchy_c, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		for (int i=0;i<(int)contours_c.size();i++)
			drawContours(srcImage, contours_c, i, Scalar(255), 1, 8);

		for (int i=0;i<(int)contours_c.size();i++)
			double g_dConArea = contourArea(contours_c[i], true);

		cvtColor(srcImage, srcImage, COLOR_GRAY2BGR);

		removeDetectionBorder(srcImage, 10);

		point_E = trackDropletBoundary(srcImage);

		for(int i=0;i<5;i++)
			outfile<<point_E[i]<<",";
		outfile<<endl;

		addWeighted(srcImage, 1, srcImage2, 1, 0.0, srcImage3);

		cout << "Complete " << k << endl;
		imwrite(str3.str(), srcImage3);
	}
	outfile.close();

	waitKey(0);

	return 0;
}
