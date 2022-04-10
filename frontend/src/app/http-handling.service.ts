import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map, retry } from 'rxjs/operators';

export interface Resp {
  lyrics: string;
  melody: string;
}

@Injectable({
  providedIn: 'root'
})

export class HttpHandlingService {
  mainUrl: string;
  constructor(private http: HttpClient) {
    this.mainUrl = "http://127.0.0.1:5000/";
  }

  async promisePostResponse(url: string, formData: FormData): Promise<Resp>{
    return new Promise((resolve, reject) => {
      try {
        let upload$ = this.http.post<Resp>(url, formData).subscribe(
          (data) => {
            data.melody = this.mainUrl + data.melody
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
    const url = this.mainUrl + "upload";
    let respData = await this.promisePostResponse(url, formData);
    return respData;
  }

  // http get methods takes in 2 parameters:
  // 1) endpoint URl from which to fetch
  // 2) option object used to configure the request

}
