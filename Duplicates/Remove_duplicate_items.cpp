#include<iostream>
#include<string.h>
using namespace std;

int main()
{
    int n,count=0;
    cin>>n;
    char a[n][80];
    for(int i=0;i<n;i++)
        cin>>a[i];

    for(int i=0;i<n;i++)
    {
        int is_duplicate =0;

        for(int j=i-1;j>=0;j--)
        {   
            if(strcmp(a[i],a[j])==0)
            {   is_duplicate = 1;
                //cout<<j+2<<" "<<i+2<<"\n";
                break;
            }
        }
        if(is_duplicate == 0)
        {
            cout<<a[i]<<"\n";
            count++;
        }
    }
    cout<<count<<"\n";
}