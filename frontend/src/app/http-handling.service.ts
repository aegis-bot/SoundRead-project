import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map, retry } from 'rxjs/operators';

export interface Resp { 
  lyrics: string;
  melody: any;
}

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

  async promisePostResponse(url: string, formData: FormData): Promise<Resp>{
    return new Promise((resolve, reject) => {
      try {
        let responseData: Resp;

        const httpOptions = {
          headers: new HttpHeaders({
            lyrics:  'lyrics',
            melody: 'melody'
          })
        };
      
        let upload$ = this.http.post<Resp>(url, formData, httpOptions).subscribe(
          (data) => {
            console.log(data);
            resolve(data);
          }
        );
      
      } catch (Error) {
          reject("Bad HTTP response");
      }
    }
    );
  }

  async receivePostResponse(url: string, formData: FormData, respData: Resp) : Promise<Resp> {
    respData = await this.promisePostResponse(url, formData);
    return respData;
  }

  

  async sendFiles(file: File): Promise<Resp> {
    console.log("sendfiles")
    const formData = new FormData();
    formData.append("fileObject", file);
    const url = "http://127.0.0.1:8000/upload/";
    let respData = await this.promisePostResponse(url, formData);
    return respData;
    /*
    let upload$ = this.http.post<Resp>("http://127.0.0.1:8000/upload/", formData).subscribe((data : Resp)=> {
      respData = {
        lyrics: data.lyrics,
        melody: data.melody
      };
    });
    */
    //let test$ = this.http.get<Resp>("dasdas")
    
  }
  
  // http get methods takes in 2 parameters:
  // 1) endpoint URl from which to fetch 
  // 2) option object used to configure the request
  
}
