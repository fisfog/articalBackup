---
title: 一个C写的简易聊天室
date: 2016-10-28
tags: 
	- Linux
	- C
categories: programing
---

近期有些空闲，正好趁着这段时间做些练手的项目巩固一些Linux下C编程:)

# 技术
	- TCP Socket编程
	- SYS V消息队列

# 设计
几乎没有设计。。简单的服务端/客户端设计，使用TCP传输数据，消息队列做进程间通信。

## Server/Client

*server端*
	- 主进程监听，子进程处理每个客户端
	- 子进程的子进程收其他进程消息发送给当前客户端

*client端*
	- 两个进程，一个进程获得终端输入，一个进程收socket数据输出

## 结构定义

消息队列结构
    typedef struct mqmesg{
		long mtype;
		long mlen;
		char mdata[MAXLEN];
    }message;

用户登录信息结构
    typedef struct lginfo{
		struct tm		*login_time; // 登录时间结构
		struct sockaddr_in	*cliaddr; // 客户端IPV4结构
		char			login_name[50+1]; // 用户登录名
    }loginfo;

# 模块
	- 通用模块 util.c
		- ssize_t readn(int, void *, size_t); // 循环read函数，确保收取n字节
		- ssize_t writen(int, void *, size_t); // 循环write函数，确保发送n字节
		- int sendMsg(int, void *, int); // 发送socket封装函数
		- int recvMsg(int, void *, int *); // 接收socket封装函数
		- int mqMsgSTInit(message *, char *, long, long); // 消息队列结构赋值函数
		- ssize_t sendMq(int, message *); // 发送消息队列封装函数
		- ssize_t recvMq(int, message *); // 接收消息队列封装函数
		- int tm2DateTimeStr(struct tm *, char *); // linux时间tm结构转YYYY-MM-DD HH:MM:SS字符串

	- 服务端功能模块 servfunc.c
		- int getClientCount(int); // 读取type 1消息代表的客户端数量
		- int putClientCount(int, int); // 向消息队列写入客户端数量
		- int login_serv(int, loginfo *); // 服务端登入处理函数 

	- 客户端功能模块 clifunc.c
		- int login_cli(int); // 客户端登入处理函数

# 实现

*server 端*
    #include "chatroom.h"    

    int main(int argc, char *argv[])
    {
    	struct sockaddr_in servaddr, cliaddr;
    	socklen_t cliaddr_len;
    	char buf[MAXLEN+1];
    	char buf2[MAXLEN] = {0};
    	char addr[INET_ADDRSTRLEN];
    	int listenfd,connfd;
    	int i,n,len;
    	int pid;
    	int client_count = 0;
    	int fpid = getpid();    

    	listenfd = socket(AF_INET, SOCK_STREAM, 0);
    	bzero(&servaddr, sizeof(servaddr));
    	servaddr.sin_family = AF_INET;
    	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    	servaddr.sin_port = htons(SERVPORT);    

    	bind(listenfd, (struct sockaddr *)&servaddr, sizeof(servaddr));    

    	listen(listenfd, 20);
    	printf("Accepting connect...\n");    

    	int mq_fd = msgget(IPC_PRIVATE, SVMSG_MODE | IPC_CREAT);
    	printf("msgid:%d\n",mq_fd);
    	putClientCount(mq_fd, client_count);
    	message *msg = (message *)malloc(sizeof(message));    

    	while(1){
    		cliaddr_len = sizeof(cliaddr);
    		connfd = accept(listenfd, (struct sockaddr *)&cliaddr, &cliaddr_len);
    		client_count = getClientCount(mq_fd);	
    		client_count++;
    		putClientCount(mq_fd, client_count);
    		pid = fork();
    		if(pid<0) printf("fork err\n");
    		else if(pid>0){
    			continue;
    		}else{
    			// child
    			printf("this is child process[%d]\n", getpid());
    			inet_ntop(AF_INET, &cliaddr.sin_addr, addr, sizeof(addr));
    			printf("Recieved connection form [%s] at PORT [%d]\n", addr, ntohs(cliaddr.sin_port));
    			int cliNo = client_count;
    			char welcome[100] = {0};
    			sprintf(welcome, "%s%d%s", "-----welcome to chat room, current user no: ", client_count, "------");
    			sendMsg(connfd, welcome, strlen(welcome));    
    

    			loginfo *cli_log_info = (loginfo *)malloc(sizeof(loginfo));
    			cli_log_info->cliaddr = &cliaddr;
    			login_serv(connfd, cli_log_info); // client login    

    			int pid2 = fork();
    			if(pid2<0){ printf("fork err\n"); continue;}
    			else if(pid2>0){
    				while(1){
    					memset(buf, 0x00, sizeof(buf));
    					if(recvMsg(connfd, buf, &len)<0){
    						printf("The client [%d] closed the connection.\n", getpid());
    						kill(pid2, SIGKILL);
    						break;
    					}
    					printf("CLIENT[%s]PID[%d]:LOGIN_NAME[%s]:LEN[%d]:MSG[%s]\n", addr, getpid(), cli_log_info->login_name, len, buf);
    					
    					time_t t = time(NULL);
    					struct tm *local = localtime(&t);
    					char tt[19+1] = {0};
    					tm2DateTimeStr(local, tt);
    					
    					sprintf(buf2, "(%s) %s: %s", tt, cli_log_info->login_name, buf);
    				//	strcat(buf, "[B]");
    	//				printf("DEBUG: [%d], buf[%s]\n", strlen(buf), buf);
    					client_count = getClientCount(mq_fd);
    					for(i=1;i<=client_count;i++){
    						if(i==cliNo) continue;
    						mqMsgSTInit(msg, buf2, strlen(buf2), 10000+i);
    						sendMq(mq_fd, msg);
    					}
    					putClientCount(mq_fd, client_count);
    				}
    			}else{
    				while(1){
    					mqMsgSTInit(msg, NULL, 0, 10000+cliNo);
    					if(recvMq(mq_fd, msg)<=0) continue;
    					else{
    						sendMsg(connfd, msg->mdata, msg->mlen);
    					}
    				}
    			}
    			client_count = getClientCount(mq_fd);
    			client_count--;
    			putClientCount(mq_fd, client_count);
    			close(connfd);
    			break;
    		}
    	}
    	close(listenfd);
    	free(msg);
    	if(getpid() == fpid){
    		printf("msgctl----------\n");
    		msgctl(mq_fd, IPC_RMID, NULL);
    	}    

    	return 0;
    }

*client 端*
    #include "chatroom.h"    

    int main(int argc, char *argv[])
    {
    	struct sockaddr_in servaddr;
    	char buf[MAXLEN], buf2[MAXLEN];
    	int sockfd;
    	int n,len,flag;
    	int pid;
    	char servip[15+1];
    	int ret;    

    	if(argc == 2)
    	{
    		strcpy(servip, argv[1]);
    	}else{
    		printf("USAGE: client [serverip]\n");
    		exit(0);
    	}    

    	sockfd = socket(AF_INET, SOCK_STREAM, 0);
    	bzero(&servaddr, sizeof(servaddr));
    	servaddr.sin_family = AF_INET;
    	inet_pton(AF_INET, (const char *)servip, &servaddr.sin_addr);
    	servaddr.sin_port = htons(SERVPORT);    

    	
    	ret = connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr));
    	if(!ret)
    		printf("Connect succeed!\n");
    	else{
    		printf("Cant connect to the server:%s\n", servip);
    		exit(1);
    	}    

    	// recieve welcome message
    	memset(buf, 0x00, sizeof(buf));
    	recvMsg(sockfd, buf, &len);
    	printf("%s\n", buf);    

    	// login
    	login_cli(sockfd);    

    	printf("---------------------------------------------------------------\n");
    	
    	pid = fork();
    	if(pid<0){
    		printf("fork err\n");
    		exit(1);
    	}else if(pid>0){
    		while(fgets(buf, MAXLEN, stdin)){
    			memset(&buf[strlen(buf)-1], 0, 1); // delete the last \n
    			sendMsg(sockfd, buf, strlen(buf));
    		}
    	}else{
    		while(1){
    			memset(buf2, 0x00, sizeof(buf2));
    			if(recvMsg(sockfd, buf2, &len)<0){
    				printf("Server [%s] closed connection\n", servip);
    				break;
    			}
    			printf("> %s\n", buf2);
    		}
    	}
    	close(sockfd);    

    	return 0;
    }

# 其他
	- 目前已知的几个缺陷：
		- 客户端终端输入输出在一起，有时候输入时会有输出冒出来
		- Ctrl-C kill掉服务端后，建立的消息队列没有删除