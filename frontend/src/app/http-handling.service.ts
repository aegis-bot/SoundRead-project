import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})

export class HttpHandlingService {
  mainUrl: string;
  constructor(private http: HttpClient) {  
    this.mainUrl = "http://127.0.0.1:8000/";
  }

  public simpleMessageTest() {
    console.log("fileSendingService");
    let queryParams = new HttpParams();
    let myMsg = "this is a message from the frontend!";
    queryParams = queryParams.append("message", myMsg);
    this.http.get<any>('http://127.0.0.1:8000/simpleMessage/', {params: queryParams, observe: 'body', responseType: 'json'}).subscribe(data => {
      console.log(data.backendMessage);
    });

  }

  

  sendFiles(file: File) {
    console.log("sendfiles")
    const formData = new FormData();
    formData.append("fileObject", file);
    const upload$ = this.http.post("http://127.0.0.1:8000/upload/", formData);
    upload$.subscribe(data=> {
      console.log(data);
    });
  }
  
  // http get methods takes in 2 parameters:
  // 1) endpoint URl from which to fetch 
  // 2) option object used to configure the request
  
}
